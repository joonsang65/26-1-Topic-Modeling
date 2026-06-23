"""
BTM 1단계: 최적 토픽 수(K) 탐색 (정밀 지표 버전)
=================================================
Gensim의 CoherenceModel을 사용하여 C_umass, C_v, NPMI를 정확하게 계산합니다.
또한 bitermplus의 int32 인덱스 오류를 해결하고 병렬 처리를 지원합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import sparse
import bitermplus as btm
from joblib import Parallel, delayed
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 utils 임포트 가능하게 함
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.data_io import parse_token_cell, get_best_text_column
from src.utils.viz_utils import set_korean_font

# 한글 폰트 설정
set_korean_font()

# ─────────────────────────────────────────────
# 0. 설정값
# ─────────────────────────────────────────────
DATA_DIR    = os.path.join("data", "processed")
TARGET_FILE = os.path.join(DATA_DIR, "최종정제_v2.csv")
OUTPUT_DIR  = os.path.join("results", "modeling_results", "BTM", "step1_k_search")
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_MIN, K_MAX = 2, 10
N_ITER = 200
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 1. 데이터 로더
# ─────────────────────────────────────────────
def load_documents():
    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {TARGET_FILE}")
    df = pd.read_csv(TARGET_FILE, encoding="utf-8-sig")
    col = get_best_text_column(df)
    docs = [parse_token_cell(v) for v in df[col]]
    return [d for d in docs if len(d) >= 2]

# ─────────────────────────────────────────────
# 2. 정밀 지표 계산 함수 (Gensim 활용)
# ─────────────────────────────────────────────
def get_accurate_coherence(model, docs, dictionary, metric='c_v'):
    """Gensim을 사용하여 수학적으로 정확한 Coherence 계산"""
    phi = model.matrix_topics_words_
    vocab = model.vocabulary_
    
    # 각 토픽별 상위 10개 단어 리스트 생성
    top_words = []
    for k in range(phi.shape[0]):
        top_ids = np.argsort(phi[k, :])[-10:][::-1]
        top_words.append([vocab[wid] for wid in top_ids])
    
    # Gensim CoherenceModel 실행
    cm = CoherenceModel(topics=top_words, texts=docs, dictionary=dictionary, coherence=metric, processes=1)
    return float(cm.get_coherence())


def calculate_biterm_perplexity(model, biterms, eps=1e-300):
    """학습된 BTM의 비터름 likelihood로 perplexity를 직접 계산.

    bitermplus의 model.perplexity_가 일부 환경에서 1e300 sentinel 값으로
    고정되는 경우가 있어 K 비교용 지표를 별도로 산출한다.
    """
    phi = model.matrix_topics_words_
    theta = model.theta_

    log_likelihood = 0.0
    n_biterms = 0

    for doc_biterms in biterms:
        for w1, w2 in doc_biterms:
            prob = np.sum(theta * phi[:, w1] * phi[:, w2])
            log_likelihood += np.log(max(float(prob), eps))
            n_biterms += 1

    if n_biterms == 0:
        return np.nan

    return float(np.exp(-log_likelihood / n_biterms))

# ─────────────────────────────────────────────
# 3. 메인 실행
# ─────────────────────────────────────────────
def main():
    # 재현성을 위한 시드 고정
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"[로드] {TARGET_FILE}")
    docs_filtered = load_documents()
    
    # Gensim Dictionary 생성 (한 번만 수행)
    dictionary = Dictionary(docs_filtered)
    
    # bitermplus용 어휘집 및 데이터 준비
    vocab = {w: i for i, w in enumerate(sorted(dictionary.token2id.keys()))}
    vocab_list = np.array([None] * len(vocab), dtype=object)
    for w, i in vocab.items(): vocab_list[i] = w
    
    docs_idx = [[vocab[w] for w in doc] for doc in docs_filtered]
    
    # ── CSR Matrix 생성 및 Dtype 고정 (ValueError: Buffer dtype mismatch 해결)
    print("문서-단어 행렬(DTM) 생성 중...")
    rows, cols, data = [], [], []
    for i, doc in enumerate(docs_idx):
        for word_id in doc:
            rows.append(i); cols.append(word_id); data.append(1)
    
    X = sparse.csr_matrix(
        (np.array(data, dtype=np.int32), 
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(len(docs_idx), len(vocab))
    )
    # 핵심 수정: 인덱스 타입을 강제로 int32로 변환
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)
    
    biterms = btm.get_biterms(docs_idx)

    k_values = list(range(K_MIN, K_MAX + 1))
    print(f"\n[ K 탐색 시작 ] 범위: {K_MIN} ~ {K_MAX} (Gensim 정밀 지표 및 병렬 처리 적용)")

    def process_k(k):
        # 모델 학습
        m = btm.BTM(X, vocab_list, T=k, M=20, alpha=0.1, beta=0.01, seed=RANDOM_SEED)
        m.fit(biterms, iterations=N_ITER)
        
        # 정확한 지표 산출
        u = get_accurate_coherence(m, docs_filtered, dictionary, 'u_mass')
        v = get_accurate_coherence(m, docs_filtered, dictionary, 'c_v')
        n = get_accurate_coherence(m, docs_filtered, dictionary, 'c_npmi')
        
        # Perplexity (보고용)
        # bitermplus의 m.perplexity_는 현재 환경에서 1e300으로 고정되어
        # K 비교가 불가능하므로, 학습된 비터름 likelihood로 직접 계산한다.
        p = calculate_biterm_perplexity(m, biterms)
        
        return {'K': k, 'C_umass': u, 'C_v': v, 'C_npmi': n, 'Perplexity': p}

    # Parallel 실행 (n_jobs=-1로 모든 코어 활용)
    results = Parallel(n_jobs=-1)(delayed(process_k)(k) for k in tqdm(k_values))

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "BTM_K_metrics_summary.csv"), index=False, encoding="utf-8-sig")

    # ── 시각화
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(res_df['K'], res_df['C_umass'], marker='o', color='steelblue', label='C_umass (Decision)')
    ax1.plot(res_df['K'], res_df['C_v'], marker='s', linestyle='--', alpha=0.7, label='C_v (Report)')
    ax1.plot(res_df['K'], res_df['C_npmi'], marker='^', linestyle=':', alpha=0.7, label='C_npmi (Evidence)')
    ax1.set_xlabel('Number of Topics (K)', fontsize=11)
    ax1.set_ylabel('Coherence Scores', fontsize=11)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(res_df['K'], res_df['Perplexity'], marker='x', color='tomato', alpha=0.5, label='Perplexity')
    ax2.set_ylabel('Perplexity (Lower is better)', color='tomato', fontsize=11)
    ax2.legend(loc='upper right')
    
    plt.title("BTM: Accurate Multi-Metric K Analysis", fontsize=13)
    plt.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "BTM_K_analysis_plot.png"), dpi=150)
    
    print(f"\n[분석 완료] 결과 저장: {OUTPUT_DIR}")
    best_k = res_df.loc[res_df['C_umass'].idxmax(), 'K']
    print(f"[ 제안 ] C_umass 기준 최적 K는 '{best_k}' 입니다.")

if __name__ == "__main__":
    main()
