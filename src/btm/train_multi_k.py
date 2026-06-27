"""
BTM 2단계: K=3,4,5,6 자동 순차 실행
=================================================
BTM_step2_optimize_v2.py를 K값만 바꿔가며 자동으로 반복 실행합니다.
"""

import os
import ast
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
from scipy import sparse
import bitermplus as btm
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis

# ─────────────────────────────────────────────
# 설정: 실행할 K값 목록
# ─────────────────────────────────────────────
K_LIST = [4, 5]

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data", "processed")
TARGET_FILE  = os.path.join(DATA_DIR, "최종정제_v2.csv")

N_TRIALS     = 30
N_ITER_OPTIM = 100
N_ITER_FINAL = 200
RANDOM_SEED  = 42
TOP_N_WORDS  = 20

# ─────────────────────────────────────────────
# 공통 함수
# ─────────────────────────────────────────────
def _parse_token_cell(cell) -> list:
    if isinstance(cell, list):
        return [str(t).strip() for t in cell if str(t).strip()]
    s = str(cell).strip()
    if not s or s in ("nan", "None", ""):
        return []
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except:
            pass
    return s.split()

def load_documents():
    df = pd.read_csv(TARGET_FILE, encoding="utf-8-sig")
    docs = [_parse_token_cell(v) for v in df['final_text']]
    return [d for d in docs if len(d) >= 2]

def build_vocab(docs: list, min_count: int = 2):
    from collections import Counter
    word_count = Counter(w for doc in docs for w in doc)
    valid = {w for w, c in word_count.items() if c >= min_count}
    vocab = {w: i for i, w in enumerate(sorted(valid))}
    filtered = [[w for w in doc if w in vocab] for doc in docs]
    return vocab, [d for d in filtered if len(d) >= 2]

def build_btm_matrix(docs_idx, vocab_size):
    rows, cols, data = [], [], []
    for i, doc in enumerate(docs_idx):
        for word_id in doc:
            rows.append(i); cols.append(word_id); data.append(1)
    X = sparse.csr_matrix(
        (np.array(data, dtype=np.int32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(len(docs_idx), vocab_size)
    )
    X.indices = X.indices.astype(np.int32)
    X.indptr  = X.indptr.astype(np.int32)
    return X

def optimize_params(docs_filtered, vocab, vocab_list, docs_idx, SELECTED_K):
    import random
    random.seed(RANDOM_SEED)
    dictionary = Dictionary(docs_filtered)

    sample_size = min(len(docs_filtered), 10000)
    docs_sample = random.sample(docs_filtered, sample_size)
    docs_idx_sample = [[vocab[w] for w in doc] for doc in docs_sample]
    X_sample = build_btm_matrix(docs_idx_sample, len(vocab))
    biterms_sample = btm.get_biterms(docs_idx_sample)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 1.0, log=True)
        beta  = trial.suggest_float("beta",  0.0001, 0.1, log=True)
        model = btm.BTM(X_sample, vocab_list, T=SELECTED_K, M=20,
                        alpha=alpha, beta=beta, seed=RANDOM_SEED)
        model.fit(biterms_sample, iterations=N_ITER_OPTIM)
        phi = model.matrix_topics_words_
        top_words = []
        for k in range(phi.shape[0]):
            top_ids = np.argsort(phi[k, :])[-10:][::-1]
            top_words.append([vocab_list[wid] for wid in top_ids])
        cm = CoherenceModel(
            topics=top_words, texts=docs_sample,
            dictionary=dictionary, coherence='u_mass', processes=1
        )
        return cm.get_coherence()

    print(f"\n[ Optuna 최적화 시작 ] K={SELECTED_K}, Trials={N_TRIALS}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)
    print(f"[ 최적 파라미터 ] alpha={study.best_params['alpha']:.4f}, beta={study.best_params['beta']:.4f}")
    print(f"[ 최적 C_umass  ] {study.best_value:.4f}")
    return study.best_params, dictionary

def save_pyldavis(model, docs_idx, vocab, vocab_list, OUTPUT_DIR, SELECTED_K):
    phi   = model.matrix_topics_words_.astype(np.float64)
    theta = model.transform([np.array(doc, dtype=np.int32) for doc in docs_idx])
    V = len(vocab)
    term_frequency = np.zeros(V, dtype=np.int32)
    for doc in docs_idx:
        for wid in doc:
            term_frequency[wid] += 1
    vocab_terms = [vocab_list[i] for i in range(V)]
    doc_lengths = np.array([len(doc) for doc in docs_idx])
    prepared = pyLDAvis.prepare(
        topic_term_dists=phi,
        doc_topic_dists=theta,
        doc_lengths=doc_lengths,
        vocab=vocab_terms,
        term_frequency=term_frequency,
        sort_topics=False
    )
    html_path = os.path.join(OUTPUT_DIR, f"BTM_pyLDAvis_K{SELECTED_K}.html")
    pyLDAvis.save_html(prepared, html_path)
    print(f"pyLDAvis 저장: {html_path}")

def run_for_k(SELECTED_K, docs_filtered, vocab, vocab_list, docs_idx, X, biterms):
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", "BTM", f"step2_K{SELECTED_K}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id2word = {i: w for w, i in vocab.items()}

    best_params, dictionary = optimize_params(docs_filtered, vocab, vocab_list, docs_idx, SELECTED_K)
    alpha = best_params['alpha']
    beta  = best_params['beta']

    print(f"\n[ 최종 모델 학습 ] K={SELECTED_K}, Alpha={alpha:.4f}, Beta={beta:.4f}")
    model = btm.BTM(X, vocab_list, T=SELECTED_K, M=20,
                    alpha=alpha, beta=beta, seed=RANDOM_SEED)
    model.fit(biterms, iterations=N_ITER_FINAL)

    print("\n[ 결과 저장 ]")
    phi = model.matrix_topics_words_

    # 토픽 단어 CSV
    topic_rows = []
    for k in range(SELECTED_K):
        top_ids = np.argsort(phi[k, :])[-TOP_N_WORDS:][::-1]
        for rank, wid in enumerate(top_ids, 1):
            topic_rows.append({"topic": k+1, "rank": rank,
                                "word": id2word[wid], "prob": float(phi[k, wid])})
    pd.DataFrame(topic_rows).to_csv(
        os.path.join(OUTPUT_DIR, "BTM_topic_words.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 문서 토픽 CSV
    docs_vec = [np.array(doc, dtype=np.int32) for doc in docs_idx]
    theta = model.transform(docs_vec)
    doc_rows = []
    for doc_id, dist in enumerate(theta):
        row = {"doc_id": doc_id, "dominant_topic": int(np.argmax(dist)) + 1}
        for k in range(SELECTED_K):
            row[f"topic_{k+1}"] = float(dist[k])
        doc_rows.append(row)
    pd.DataFrame(doc_rows).to_csv(
        os.path.join(OUTPUT_DIR, "BTM_document_topics.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 파라미터 저장
    with open(os.path.join(OUTPUT_DIR, "BTM_final_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"K: {SELECTED_K}\nAlpha: {alpha}\nBeta: {beta}\n")
        f.write(f"N_iter_final: {N_ITER_FINAL}\nM: 20\n")

    # 토픽-단어 시각화
    fig, axes = plt.subplots(
        nrows=(SELECTED_K + 1) // 2, ncols=2,
        figsize=(14, 3 * ((SELECTED_K + 1) // 2))
    )
    axes = axes.flatten()
    for k in range(SELECTED_K):
        top_ids = np.argsort(phi[k, :])[-10:][::-1]
        words = [id2word[wid] for wid in top_ids]
        probs = [float(phi[k, wid]) for wid in top_ids]
        axes[k].barh(words[::-1], probs[::-1], color='steelblue')
        axes[k].set_title(f"Topic {k+1}", fontsize=11)
        axes[k].set_xlabel("Probability")
    for k in range(SELECTED_K, len(axes)):
        fig.delaxes(axes[k])
    plt.suptitle(f"BTM Topic-Word Distribution (K={SELECTED_K})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "BTM_topic_words_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # pyLDAvis HTML
    print("\n[ pyLDAvis 생성 중... ]")
    save_pyldavis(model, docs_idx, vocab, vocab_list, OUTPUT_DIR, SELECTED_K)

    print(f"\n[K={SELECTED_K} 완료] 결과 폴더: {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def main():
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"[로드] {TARGET_FILE}")
    docs = load_documents()
    vocab, docs_filtered = build_vocab(docs)

    vocab_list = np.array([None] * len(vocab), dtype=object)
    for w, i in vocab.items():
        vocab_list[i] = w

    docs_idx = [[vocab[w] for w in doc] for doc in docs_filtered]
    X = build_btm_matrix(docs_idx, len(vocab))
    biterms = btm.get_biterms(docs_idx)

    print(f"\n총 {len(K_LIST)}개 K값 순차 실행: {K_LIST}")

    for SELECTED_K in K_LIST:
        print(f"\n{'='*50}")
        print(f"  K = {SELECTED_K} 시작")
        print(f"{'='*50}")
        try:
            run_for_k(SELECTED_K, docs_filtered, vocab, vocab_list, docs_idx, X, biterms)
        except Exception as e:
            print(f"[오류] K={SELECTED_K} 실패: {e}")
            import traceback
            traceback.print_exc()
            print("다음 K로 계속합니다...")

    print(f"\n{'='*50}")
    print(f"전체 완료! K={K_LIST} 모두 처리됨")
    print(f"결과 위치: {os.path.join(BASE_DIR, 'results', 'BTM')}")

if __name__ == "__main__":
    main()
