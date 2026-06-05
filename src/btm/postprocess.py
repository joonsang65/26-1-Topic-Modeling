"""
BTM 최종 모델 생성 및 인터랙티브 CSV 제작 스크립트
=================================================
step2_K{3,4,5} 폴더의 BTM_final_params_K{k}.txt 파일을 읽어
최적 모델을 다시 학습하고, 원본 데이터와 결합된 상세 CSV를 생성합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import sparse
import bitermplus as btm
from tqdm import tqdm
import pyLDAvis
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 utils 임포트 가능하게 함
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.data_io import parse_token_cell, get_best_text_column

# ─────────────────────────────────────────────
# 0. 설정값
# ─────────────────────────────────────────────
K_LIST = [3, 4, 5]
DATA_DIR = os.path.join("data", "processed")
TARGET_FILE = os.path.join(DATA_DIR, "최종정제_v2.csv")
BASE_RESULT_DIR = os.path.join("results", "modeling_results", "BTM")

N_ITER = 200
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 1. 유틸리티 함수
# ─────────────────────────────────────────────
def load_data_with_mapping():
    """원본 데이터를 로드하고 필터링 과정을 추적하여 인덱스를 유지함"""
    print(f"[1/5] 데이터 로드 중: {TARGET_FILE}")
    df = pd.read_csv(TARGET_FILE, encoding="utf-8-sig")
    
    # 토큰화 및 1차 필터링 (길이 2 이상)
    col = get_best_text_column(df)
    df['tokens'] = df[col].apply(parse_token_cell)
    df_filtered = df[df['tokens'].apply(len) >= 2].copy()
    
    print(f"      - 전체 문서: {len(df)}")
    print(f"      - 1차 필터링 후 (길이>=2): {len(df_filtered)}")
    return df_filtered

def build_vocab_and_filter(df_filtered, min_count=2):
    """어휘 사전을 구축하고 어휘에 포함된 단어가 2개 미만인 문서를 최종 제외"""
    from collections import Counter
    print(f"[2/5] 어휘 사전 구축 중 (min_count={min_count})")
    
    all_tokens = [t for tokens in df_filtered['tokens'] for t in tokens]
    word_count = Counter(all_tokens)
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in word_count.items() if c >= min_count))}
    
    # 어휘 사전에 있는 단어들로만 재구성
    df_filtered['tokens_in_vocab'] = df_filtered['tokens'].apply(lambda x: [t for t in x if t in vocab])
    
    # 최종 필터링
    df_final = df_filtered[df_filtered['tokens_in_vocab'].apply(len) >= 2].copy()
    
    print(f"      - 어휘 크기: {len(vocab)}")
    print(f"      - 최종 분석 대상 문서: {len(df_final)}")
    return vocab, df_final

def get_top_words_dict(model, vocab, k, topn=10):
    """각 토픽별 상위 단어를 딕셔너리 형태로 반환"""
    id2word = {i: w for w, i in vocab.items()}
    phi = model.matrix_topics_words_
    top_words_map = {}
    for t in range(k):
        top_ids = np.argsort(phi[t, :])[-topn:][::-1]
        words = [id2word[wid] for wid in top_ids]
        top_words_map[t+1] = ", ".join(words)
    return top_words_map

def save_btm_pyldavis(model, theta, X, vocab_list, save_path):
    """BTM 결과를 pyLDAvis HTML로 저장"""
    try:
        # bitermplus 0.10.0 이상에서는 matrix_topics_words_ (phi) 사용
        phi = model.matrix_topics_words_
        
        # doc_lengths: 각 문서의 단어 수 (X의 행별 합)
        doc_lengths = np.array(X.sum(axis=1)).flatten()
        
        # term_frequency: 각 단어의 총 빈도 (X의 열별 합)
        term_frequency = np.array(X.sum(axis=0)).flatten()
        
        # pyLDAvis 데이터 준비
        vis_data = pyLDAvis.prepare(
            topic_term_dists = phi,
            doc_topic_dists   = theta,
            doc_lengths       = doc_lengths,
            vocab             = vocab_list,
            term_frequency    = term_frequency,
            mds               = 'mmds'  # 또는 'tsne'
        )
        pyLDAvis.save_html(vis_data, save_path)
        print(f"      - 시각화 저장 완료: {save_path}")
    except Exception as e:
        print(f"      - [경고] 시각화 생성 실패: {e}")

# ─────────────────────────────────────────────
# 2. 메인 실행 로직
# ─────────────────────────────────────────────
def main():
    np.random.seed(RANDOM_SEED)
    
    # 데이터 준비
    df_filtered = load_data_with_mapping()
    vocab, df_final = build_vocab_and_filter(df_filtered)
    
    # BTM 학습을 위한 입력 데이터 생성
    docs_idx = [[vocab[w] for w in doc] for doc in df_final['tokens_in_vocab']]
    vocab_list = np.array([None] * len(vocab), dtype=object)
    for w, i in vocab.items(): vocab_list[i] = w
    
    rows, cols, data = [], [], []
    for i, doc in enumerate(docs_idx):
        for word_id in doc:
            rows.append(i); cols.append(word_id); data.append(1)
            
    X = sparse.csr_matrix(
        (np.array(data, dtype=np.int32), 
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))), 
        shape=(len(docs_idx), len(vocab))
    )
    biterms = btm.get_biterms(docs_idx)

    for k in K_LIST:
        TARGET_DIR = os.path.join(BASE_RESULT_DIR, f"step2_K{k}")
        PARAM_FILE = os.path.join(TARGET_DIR, f"BTM_final_params_K{k}.txt")
        
        if not os.path.exists(PARAM_FILE):
            print(f"\n[경고] {PARAM_FILE} 파일을 찾을 수 없어 K={k} 건너뜁니다.")
            continue
            
        # 파라미터 읽기
        params = {}
        with open(PARAM_FILE, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":")
                    params[key.strip().lower()] = float(val.strip())
        
        alpha = params.get('alpha', 0.1)
        beta = params.get('beta', 0.01)
        
        print(f"\n[3/5] 최적 모델 학습 (K={k}, alpha={alpha:.4f}, beta={beta:.4f})")
        model = btm.BTM(X, vocab_list, T=k, M=20, alpha=alpha, beta=beta, seed=RANDOM_SEED)
        model.fit(biterms, iterations=N_ITER)
        
        # 문서-토픽 분포 계산
        docs_vec = [np.array(doc, dtype=np.int32) for doc in docs_idx]
        theta = model.transform(docs_vec)
        
        # ─────────────────────────────────────────────
        # 4. 인터랙티브 CSV 생성 (결과 병합)
        # ─────────────────────────────────────────────
        print(f"[4/5] 인터랙티브 CSV 생성 중 (K={k})")
        
        # 토픽별 상위 단어 정보
        top_words_map = get_top_words_dict(model, vocab, k, topn=15)
        
        # 결과 DataFrame 복사
        res_df = df_final.copy()
        
        # 토픽 확률 및 지배 토픽 추가
        res_df['dominant_topic'] = np.argmax(theta, axis=1) + 1
        res_df['dominant_topic_prob'] = np.max(theta, axis=1)
        
        for t in range(k):
            res_df[f'topic_{t+1}_prob'] = theta[:, t]
            
        # 지배 토픽의 상위 단어 리스트 추가
        res_df['topic_keywords'] = res_df['dominant_topic'].map(top_words_map)
        
        # 불필요한 컬럼 정리 (토큰 컬럼 등)
        cols_to_keep = ['cid', 'final_text', 'dominant_topic', 'dominant_topic_prob', 'topic_keywords'] + \
                       [f'topic_{t+1}_prob' for t in range(k)]
        interactive_df = res_df[cols_to_keep]
        
        # 저장
        save_csv_path = os.path.join(TARGET_DIR, f"BTM_interactive_result_K{k}.csv")
        interactive_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"      - CSV 저장 완료: {save_csv_path}")

        # ─────────────────────────────────────────────
        # 5. 인터랙티브 HTML 시각화 (pyLDAvis)
        # ─────────────────────────────────────────────
        print(f"[5/5] 인터랙티브 HTML 생성 중 (K={k})")
        save_html_path = os.path.join(TARGET_DIR, f"BTM_pyLDAvis_K{k}.html")
        save_btm_pyldavis(model, theta, X, vocab_list, save_html_path)

    print("\n[ 모든 작업이 성공적으로 완료되었습니다. ]")

if __name__ == "__main__":
    main()
