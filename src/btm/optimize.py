"""
BTM 2단계: 파라미터 최적화 및 최종 학습
=================================================
1단계에서 결정된 K값을 바탕으로 Optuna를 사용하여 alpha, beta를 최적화하고
최종 모델을 학습하여 결과를 저장합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from tqdm import tqdm
from scipy import sparse
import bitermplus as btm
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
# 0. 설정값 (K값 리스트를 입력하세요)
# ─────────────────────────────────────────────
K_LIST = [3, 4, 5]  # 분석하고 싶은 K값들을 리스트로 입력

DATA_DIR    = os.path.join("data", "processed")
TARGET_FILE = os.path.join(DATA_DIR, "최종정제_v2.csv")
BASE_OUTPUT_DIR = os.path.join("results", "modeling_results", "BTM")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

N_TRIALS = 30       # [30]
N_ITER = 200        # [200]
N_ITER_OPTIM = 100  # [100]
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 1. 데이터 로더 (기존 로직 유지)
# ─────────────────────────────────────────────
def load_documents():
    df = pd.read_csv(TARGET_FILE, encoding="utf-8-sig")
    col = get_best_text_column(df)
    docs = [parse_token_cell(v) for v in df[col]]
    return [d for d in docs if len(d) >= 2]

def build_vocab(docs: list, min_count: int = 2):
    from collections import Counter
    word_count = Counter(w for doc in docs for w in doc)
    valid = {w for w, c in word_count.items() if c >= min_count}
    vocab = {w: i for i, w in enumerate(sorted(valid))}
    filtered = [[w for w in doc if w in vocab] for doc in docs]
    return vocab, [d for d in filtered if len(d) >= 2]

# ─────────────────────────────────────────────
# 2. Optuna 최적화
# ─────────────────────────────────────────────

def custom_coherence(topic_words, X, M=10):
    import numpy as np
    K = topic_words.shape[0]
    doc_count = X.shape[0]
    
    # 단어 출현 횟수가 아닌 '출현 여부(1 or 0)'로 변환
    X_bin = (X > 0).astype(np.int32)
    
    # 총 짝(Pair)의 개수 계산: M개 중 2개를 고르는 조합의 수 (M=10이면 45개)
    num_pairs = M * (M - 1) / 2
    
    coherence_scores = []
    for k in range(K):
        # 해당 토픽의 상위 M개 단어 인덱스 추출
        top_words = np.argsort(topic_words[k])[-M:]
        score = 0.0
        
        # M개 단어들끼리의 모든 짝(Pair)에 대해 동시 등장 확률(PMI) 계산
        for i in range(M - 1):
            for j in range(i + 1, M):
                w1, w2 = top_words[i], top_words[j]
                
                col1 = X_bin[:, w1].toarray().flatten()
                col2 = X_bin[:, w2].toarray().flatten()
                
                df1 = col1.sum()
                df2 = col2.sum()
                co_occur = (col1 * col2).sum()
                
                # 두 단어가 동시에 등장한 적이 있다면 PMI 점수 합산
                if df1 > 0 and df2 > 0 and co_occur > 0:
                    pmi = np.log((co_occur * doc_count) / (df1 * df2))
                    score += pmi
                    
        # [수정된 부분] 토픽 내의 합산 점수를 총 짝의 개수(45)로 나누어 '평균'을 구함
        topic_avg_score = score / num_pairs if num_pairs > 0 else 0
        coherence_scores.append(topic_avg_score)
        
    # 모든 토픽들의 평균 점수들을 다시 전체 평균 내어 최종 반환
    return np.mean(coherence_scores)

def optimize_params(docs_filtered, vocab, k):
    import random
    random.seed(RANDOM_SEED)
    
    # 속도 향상을 위한 샘플링 (1만 건)
    sample_size = min(len(docs_filtered), 10000)
    docs_sample = random.sample(docs_filtered, sample_size)
    docs_idx_sample = [[vocab[w] for w in doc] for doc in docs_sample]
    
    vocab_list = np.array([None] * len(vocab), dtype=object)
    for w, i in vocab.items(): vocab_list[i] = w

    # 준비
    rows, cols, data = [], [], []
    for i, doc in enumerate(docs_idx_sample):
        for word_id in doc: rows.append(i); cols.append(word_id); data.append(1)
    
    # 데이터는 float64, 인덱스는 int32로 설정합니다.
    X_sample = sparse.csr_matrix(
        (np.array(data, dtype=np.int32), 
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))), 
        shape=(len(docs_idx_sample), len(vocab))
    )
    X_sample.indices = X_sample.indices.astype(np.int32)
    X_sample.indptr = X_sample.indptr.astype(np.int32)
    
    biterms_sample = btm.get_biterms(docs_idx_sample)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
        beta  = trial.suggest_float("beta", 0.0001, 0.1, log=True)

        model = btm.BTM(X_sample, vocab_list, T=k, M=20, alpha=alpha, beta=beta, seed=RANDOM_SEED)
        model.fit(biterms_sample, iterations=N_ITER_OPTIM)
        
        # 확률 행렬은 64비트 실수로 고정
        matrix_fixed = model.matrix_topics_words_.astype(np.float64)
        
        # X_sample은 처음부터 32비트 정수로 완벽하게 만들었으므로 astype() 없이 그대로 넣습니다!
        coh = custom_coherence(matrix_fixed, X_sample, M=10)
        
        return coh - (alpha * 1e-5) - (beta * 1e-4)

    print(f"\n[ Optuna 최적화 시작 ] K={k}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)
    
    return study.best_params

# ─────────────────────────────────────────────
# 3. 메인 실행
# ─────────────────────────────────────────────
def main():
    # 재현성을 위한 시드 고정
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    docs = load_documents()
    vocab, docs_filtered = build_vocab(docs)
    
    for k in K_LIST:
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"step2_K{k}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # ── 1. 파라미터 최적화
        best_params = optimize_params(docs_filtered, vocab, k)
        alpha, beta = best_params['alpha'], best_params['beta']
        
        # ── 2. 최종 모델 학습 (전체 데이터)
        print(f"\n[ 최종 모델 학습 ] K={k}, Alpha={alpha:.4f}, Beta={beta:.4f}")
        docs_idx = [[vocab[w] for w in doc] for doc in docs_filtered]
        rows, cols, data = [], [], []
        for i, doc in enumerate(docs_idx):
            for word_id in doc: rows.append(i); cols.append(word_id); data.append(1)
        
        # 데이터는 float64, 인덱스는 int32로 설정합니다.
        X = sparse.csr_matrix(
            (np.array(data, dtype=np.int32), 
             (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))), 
            shape=(len(docs_idx), len(vocab))
        )
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int32)
        
        biterms = btm.get_biterms(docs_idx)
        
        vocab_list = np.array([None] * len(vocab), dtype=object)
        for w, i in vocab.items(): vocab_list[i] = w

        model = btm.BTM(X, vocab_list, T=k, M=20, alpha=alpha, beta=beta, seed=RANDOM_SEED)
        model.fit(biterms, iterations=N_ITER)

        # ── 3. 결과 저장
        print(f"\n[ 결과 저장 ] K={k}")
        # 토픽 단어 저장
        phi = model.matrix_topics_words_
        id2word = {i: w for w, i in vocab.items()}
        topic_rows = []
        for t in range(k):
            top_ids = np.argsort(phi[t, :])[-20:][::-1]
            for rank, wid in enumerate(top_ids, 1):
                topic_rows.append({"topic": t+1, "rank": rank, "word": id2word[wid], "prob": phi[t, wid]})
        pd.DataFrame(topic_rows).to_csv(os.path.join(OUTPUT_DIR, f"BTM_topic_words_K{k}.csv"), index=False, encoding="utf-8-sig")

        # 문서 토픽 저장
        docs_vec = [np.array(doc, dtype=np.int32) for doc in docs_idx]
        theta = model.transform(docs_vec)
        doc_rows = []
        for doc_id, dist in enumerate(theta):
            row = {"doc_id": doc_id, "dominant_topic": np.argmax(dist) + 1}
            for t in range(k): row[f"topic_{t+1}"] = dist[t]
            doc_rows.append(row)
        pd.DataFrame(doc_rows).to_csv(os.path.join(OUTPUT_DIR, f"BTM_document_topics_K{k}.csv"), index=False, encoding="utf-8-sig")

        # 설정값 저장
        with open(os.path.join(OUTPUT_DIR, f"BTM_final_params_K{k}.txt"), "w") as f:
            f.write(f"K: {k}\nAlpha: {alpha}\nBeta: {beta}\n")

        print(f"K={k} 완료! 결과 폴더: {OUTPUT_DIR}")

    print("\n[ 모든 K에 대해 작업이 완료되었습니다. ]")

if __name__ == "__main__":
    main()