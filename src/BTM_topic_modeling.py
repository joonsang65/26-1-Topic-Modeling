"""
BTM 토픽 모델링 - 최적 K값 탐색 포함
=================================================
Biterm Topic Model (BTM) 은 문서가 짧을 때 효과적인 토픽 모델입니다.
(뉴스 헤드라인, SNS 게시물, 리뷰 등 단문 텍스트에 특히 유리)

전처리_정리 폴더의 전처리된 텍스트 파일을 읽어 BTM을 수행합니다.

설치 패키지:
  pip install bitermplus matplotlib seaborn pandas openpyxl tqdm numpy scipy

주의: bitermplus 는 C 확장을 사용합니다.
  - Windows: pip install bitermplus --no-build-isolation
  - 오류 시: pip install bitermplus==0.12.3
"""

import os
import re
import ast
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 설정값 (필요에 따라 수정)
# ─────────────────────────────────────────────
DATA_DIR    = os.path.join("data", "processed")
OUTPUT_DIR  = os.path.join("results", "modeling_results", "BTM")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# K 탐색 범위
K_START     = 2
K_END       = 15
K_STEP      = 1

# BTM 하이퍼파라미터
ALPHA       = 1.0      # 토픽 분포 디리클레 파라미터 (문서-토픽)
BETA        = 0.01     # 단어 분포 디리클레 파라미터 (토픽-단어)
N_ITER      = 200      # 깁스 샘플링 반복 수

# 한글 폰트 설정
try:
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = font_prop.get_name()
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────
# 1. 데이터 로드 (LDA와 동일한 로더)
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
        except Exception:
            pass
    return s.split()


def _best_text_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        sample  = df[col].dropna().head(5)
        lengths = sample.apply(lambda x: len(_parse_token_cell(x)))
        if lengths.mean() >= 2:
            candidates.append((col, lengths.mean()))
    if not candidates:
        raise ValueError("토큰화된 텍스트 컬럼을 찾지 못했습니다.")
    candidates.sort(key=lambda x: -x[1])
    print(f"  → 텍스트 컬럼 자동 선택: '{candidates[0][0]}'")
    return candidates[0][0]


def load_documents(data_dir: str) -> list:
    all_docs = []
    files = [f for f in os.listdir(data_dir)
             if f.lower().endswith((".csv", ".txt", ".xlsx", ".xls"))]
    if not files:
        raise FileNotFoundError(f"'{data_dir}' 에 csv/txt/xlsx 파일이 없습니다.")

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        ext   = fname.lower().rsplit(".", 1)[-1]
        print(f"[로드] {fname}")

        if ext == "txt":
            with open(fpath, "r", encoding="utf-8") as f:
                docs = [line.strip().split() for line in f if line.strip()]
        elif ext == "csv":
            df   = pd.read_csv(fpath, encoding="utf-8-sig")
            col  = _best_text_column(df)
            docs = [_parse_token_cell(v) for v in df[col]]
        else:
            df   = pd.read_excel(fpath)
            col  = _best_text_column(df)
            docs = [_parse_token_cell(v) for v in df[col]]

        docs = [d for d in docs if len(d) >= 2]
        print(f"     문서 수: {len(docs)}")
        all_docs.extend(docs)

    print(f"\n총 문서 수: {len(all_docs)}\n")
    return all_docs


# ─────────────────────────────────────────────
# 2. 어휘 구축 (BTM용)
# ─────────────────────────────────────────────
def build_vocab(docs: list, min_count: int = 2, max_ratio: float = 0.95):
    """
    단어 빈도 기반 어휘 사전 구축.
    min_count  : 최소 등장 횟수
    max_ratio  : 전체 문서 중 등장 비율 상한
    반환: (vocab dict {word: id}, filtered_docs)
    """
    from collections import Counter
    word_count = Counter(w for doc in docs for w in doc)
    n_docs     = len(docs)

    # 문서 빈도 계산
    doc_freq = Counter()
    for doc in docs:
        for w in set(doc):
            doc_freq[w] += 1

    # 필터
    valid = {w for w, c in word_count.items()
             if c >= min_count and doc_freq[w] / n_docs <= max_ratio}

    vocab = {w: i for i, w in enumerate(sorted(valid))}
    filtered = [[w for w in doc if w in vocab] for doc in docs]
    filtered = [d for d in filtered if len(d) >= 2]

    print(f"어휘 크기: {len(vocab)},  유효 문서 수: {len(filtered)}")
    return vocab, filtered


# ─────────────────────────────────────────────
# 3. BTM 모델 학습 (bitermplus 사용)
# ─────────────────────────────────────────────
def train_btm(docs_filtered: list, vocab: dict, num_topics: int,
              alpha=ALPHA, beta=BETA, n_iter=N_ITER):
    """
    bitermplus 로 BTM 학습.
    반환: (btm_model, biterms)
    """
    import bitermplus as btm

    # 어휘 id로 변환된 문서
    docs_idx = [[vocab[w] for w in doc if w in vocab]
                for doc in docs_filtered]

    vocab_list = [None] * len(vocab)
    for w, i in vocab.items():
        vocab_list[i] = w

    # BTM 인스턴스 생성 및 학습
    model = btm.BTM(
        X        = docs_idx,       # 문서 리스트 (단어 id)
        vocabulary = vocab_list,   # 어휘 리스트
        T        = num_topics,     # 토픽 수
        M        = 20,             # 단어 pair 샘플링 윈도우
        alpha    = alpha,
        beta     = beta,
    )
    model.fit(docs_idx, iterations=n_iter)
    return model, docs_idx


# ─────────────────────────────────────────────
# 4. 최적 K 탐색 - Perplexity & Topic Coherence
# ─────────────────────────────────────────────
def _topic_coherence_btm(model, docs_idx: list, vocab: dict,
                         topn: int = 10) -> float:
    """
    BTM 모델의 토픽 코히런스 계산 (UCI Pointwise MI 근사).
    공식: Coherence = mean over topics of mean PMI(w_i, w_j)
    """
    import bitermplus as btm
    phi = model.matrix_words_topics_  # shape: (vocab, K)

    # 단어 공출현 빈도 계산
    doc_sets = [set(d) for d in docs_idx]
    n_docs   = len(doc_sets)

    word_freq  = {}
    cooc_freq  = {}
    for d in doc_sets:
        d = list(d)
        for w in d:
            word_freq[w] = word_freq.get(w, 0) + 1
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                pair = (min(d[i], d[j]), max(d[i], d[j]))
                cooc_freq[pair] = cooc_freq.get(pair, 0) + 1

    K = phi.shape[1]
    coherences = []
    for k in range(K):
        top_words = np.argsort(phi[:, k])[-topn:][::-1].tolist()
        pmi_scores = []
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                wi, wj = top_words[i], top_words[j]
                pair   = (min(wi, wj), max(wi, wj))
                c_ij   = cooc_freq.get(pair, 0) + 1  # 스무딩
                c_i    = word_freq.get(wi, 0)  + 1
                c_j    = word_freq.get(wj, 0)  + 1
                pmi    = np.log((c_ij * n_docs) / (c_i * c_j + 1e-12) + 1e-12)
                pmi_scores.append(pmi)
        coherences.append(np.mean(pmi_scores) if pmi_scores else 0.0)
    return float(np.mean(coherences))


def search_optimal_k_btm(docs_filtered: list, vocab: dict,
                          k_start=K_START, k_end=K_END, k_step=K_STEP,
                          alpha=ALPHA, beta=BETA, n_iter=N_ITER):
    """
    K를 변화시키며 Perplexity & Coherence 측정.
    반환: (k_list, perplexity_list, coherence_list, best_k)
    """
    import bitermplus as btm

    docs_idx = [[vocab[w] for w in doc if w in vocab]
                for doc in docs_filtered]
    vocab_list = [None] * len(vocab)
    for w, i in vocab.items():
        vocab_list[i] = w

    k_list      = list(range(k_start, k_end + 1, k_step))
    perplexities = []
    coherences   = []

    print(f"\n[ BTM K 탐색 ] 범위: {k_start} ~ {k_end}")
    for k in tqdm(k_list, desc="BTM K 탐색"):
        model = btm.BTM(
            X         = docs_idx,
            vocabulary= vocab_list,
            T         = k,
            M         = 20,
            alpha     = alpha,
            beta      = beta,
        )
        model.fit(docs_idx, iterations=n_iter)

        # Perplexity (낮을수록 좋음)
        try:
            ppl = model.perplexity_
        except Exception:
            ppl = np.nan

        # Coherence (높을수록 좋음)
        coh = _topic_coherence_btm(model, docs_idx, vocab, topn=10)

        perplexities.append(ppl)
        coherences.append(coh)

    # ── 최적 K: Coherence 최대
    valid_coh  = [(i, c) for i, c in enumerate(coherences) if not np.isnan(c)]
    best_idx   = max(valid_coh, key=lambda x: x[1])[0]
    best_k     = k_list[best_idx]

    print(f"\nK별 결과:")
    for k, ppl, coh in zip(k_list, perplexities, coherences):
        marker = " ← 최적(Coherence)" if k == best_k else ""
        print(f"  K={k:2d}  Perplexity={ppl:.2f}  Coherence={coh:.4f}{marker}")

    # ── 시각화 저장
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Perplexity
    valid_ppl = [(k_list[i], v) for i, v in enumerate(perplexities)
                 if not np.isnan(v)]
    if valid_ppl:
        kk, vv = zip(*valid_ppl)
        axes[0].plot(kk, vv, marker="o", color="tomato")
        axes[0].set_xlabel("토픽 수 (K)", fontsize=11)
        axes[0].set_ylabel("Perplexity", fontsize=11)
        axes[0].set_title("BTM: K별 Perplexity", fontsize=12)
        axes[0].grid(True, alpha=0.3)

    # Coherence
    axes[1].plot(k_list, coherences, marker="o", color="steelblue")
    axes[1].axvline(best_k, color="tomato", linestyle="--",
                    label=f"Best K={best_k}")
    axes[1].set_xlabel("토픽 수 (K)", fontsize=11)
    axes[1].set_ylabel("Coherence (PMI)", fontsize=11)
    axes[1].set_title("BTM: K별 Coherence", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "BTM_K_search.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  → 저장: {save_path}")

    return k_list, perplexities, coherences, best_k


# ─────────────────────────────────────────────
# 5. 결과 저장
# ─────────────────────────────────────────────
def save_btm_topic_words(model, vocab: dict, best_k: int, topn: int = 20):
    """토픽별 상위 단어를 CSV로 저장"""
    id2word = {i: w for w, i in vocab.items()}
    phi     = model.matrix_words_topics_   # (vocab_size, K)

    rows = []
    for k in range(best_k):
        top_ids = np.argsort(phi[:, k])[-topn:][::-1]
        for rank, wid in enumerate(top_ids, 1):
            rows.append({
                "topic":       k + 1,
                "rank":        rank,
                "word":        id2word.get(wid, ""),
                "probability": round(float(phi[wid, k]), 6),
            })
    df = pd.DataFrame(rows)
    save_path = os.path.join(OUTPUT_DIR, "BTM_topic_words.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {save_path}")
    return df


def save_btm_document_topics(model, docs_idx: list, best_k: int):
    """각 문서의 토픽 분포를 CSV로 저장"""
    import bitermplus as btm

    # 문서-토픽 분포 추론
    theta = model.transform(docs_idx)   # shape: (n_docs, K)

    rows = []
    for doc_id, dist in enumerate(theta):
        row = {"doc_id": doc_id}
        for k in range(best_k):
            row[f"topic_{k+1}"] = round(float(dist[k]), 6)
        row["dominant_topic"] = int(np.argmax(dist)) + 1
        rows.append(row)
    df = pd.DataFrame(rows)
    save_path = os.path.join(OUTPUT_DIR, "BTM_document_topics.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {save_path}")
    return df


def plot_btm_topic_word_heatmap(model, vocab: dict, best_k: int, topn: int = 10):
    """토픽-단어 확률 히트맵 저장"""
    id2word = {i: w for w, i in vocab.items()}
    phi     = model.matrix_words_topics_   # (vocab_size, K)

    data, words_all = {}, []
    for k in range(best_k):
        top_ids = np.argsort(phi[:, k])[-topn:][::-1]
        data[f"T{k+1}"] = {id2word[wid]: float(phi[wid, k]) for wid in top_ids}
        words_all.extend([id2word[wid] for wid in top_ids])
    words_all = list(dict.fromkeys(words_all))

    mat = pd.DataFrame(
        [[data.get(f"T{k+1}", {}).get(w, 0) for w in words_all]
         for k in range(best_k)],
        index=[f"T{k+1}" for k in range(best_k)],
        columns=words_all,
    )
    fig_w = max(12, len(words_all) * 0.45)
    fig_h = max(4,  best_k * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(mat, ax=ax, cmap="YlOrRd", linewidths=0.3,
                annot=False, xticklabels=True)
    ax.set_title(f"BTM 토픽-단어 히트맵 (K={best_k})", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "BTM_topic_word_heatmap.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {save_path}")


def plot_btm_topic_bar(model, vocab: dict, best_k: int, topn: int = 10):
    """토픽별 상위 단어 막대 그래프 저장"""
    id2word = {i: w for w, i in vocab.items()}
    phi     = model.matrix_words_topics_

    ncols  = min(3, best_k)
    nrows  = (best_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for k in range(best_k):
        top_ids   = np.argsort(phi[:, k])[-topn:][::-1]
        words     = [id2word[wid] for wid in top_ids]
        probs     = [float(phi[wid, k]) for wid in top_ids]
        ax = axes[k]
        bars = ax.barh(words[::-1], probs[::-1], color="steelblue")
        ax.set_title(f"Topic {k+1}", fontsize=11)
        ax.set_xlabel("확률", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

    # 남는 subplot 숨기기
    for i in range(best_k, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"BTM 토픽별 상위 단어 (K={best_k})", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "BTM_topic_top_words.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 저장: {save_path}")


def plot_btm_dominant_topic_dist(doc_topic_df: pd.DataFrame, best_k: int):
    """문서별 dominant topic 분포 막대 그래프 저장"""
    counts = doc_topic_df["dominant_topic"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(max(6, best_k * 0.8), 4))
    ax.bar([f"T{t}" for t in counts.index], counts.values, color="steelblue")
    ax.set_xlabel("토픽", fontsize=12)
    ax.set_ylabel("문서 수", fontsize=12)
    ax.set_title("BTM 토픽별 문서 수 분포", fontsize=13)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "BTM_dominant_topic_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {save_path}")


# ─────────────────────────────────────────────
# 6. 메인 파이프라인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BTM 토픽 모델링 파이프라인")
    print("=" * 60)

    # ── 1. bitermplus 존재 확인
    try:
        import bitermplus as btm
        print(f"bitermplus 버전: {btm.__version__}")
    except ImportError:
        print("[오류] bitermplus 가 설치되어 있지 않습니다.")
        print("  → pip install bitermplus")
        return

    # ── 2. 데이터 로드
    docs = load_documents(DATA_DIR)

    # ── 3. 어휘 구축
    vocab, docs_filtered = build_vocab(docs, min_count=2, max_ratio=0.95)

    # ── 4. 최적 K 탐색
    k_list, perplexities, coherences, best_k = search_optimal_k_btm(
        docs_filtered, vocab,
        k_start=K_START, k_end=K_END, k_step=K_STEP,
        alpha=ALPHA, beta=BETA, n_iter=N_ITER,
    )

    # ── 5. 최종 모델 학습
    print(f"\n[ 최종 BTM 모델 학습 ] K={best_k}")
    model, docs_idx = train_btm(
        docs_filtered, vocab, num_topics=best_k,
        alpha=ALPHA, beta=BETA, n_iter=N_ITER,
    )

    # ── 6. 결과 저장
    print("\n[ 결과 저장 ]")
    topic_word_df = save_btm_topic_words(model, vocab, best_k, topn=20)
    doc_topic_df  = save_btm_document_topics(model, docs_idx, best_k)
    plot_btm_topic_word_heatmap(model, vocab, best_k, topn=10)
    plot_btm_topic_bar(model, vocab, best_k, topn=10)
    plot_btm_dominant_topic_dist(doc_topic_df, best_k)

    # ── 7. K 탐색 요약 CSV
    summary = pd.DataFrame({
        "K":           k_list,
        "perplexity":  perplexities,
        "coherence":   coherences,
        "best_k":      [best_k] * len(k_list),
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "BTM_K_search_summary.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"  → 저장: {os.path.join(OUTPUT_DIR, 'BTM_K_search_summary.csv')}")

    # ── 8. 콘솔 요약
    id2word = {i: w for w, i in vocab.items()}
    phi     = model.matrix_words_topics_
    print("\n[ 최종 토픽 요약 ]")
    for k in range(best_k):
        top_ids = np.argsort(phi[:, k])[-10:][::-1]
        words   = [id2word[wid] for wid in top_ids]
        print(f"  Topic {k+1:2d}: {', '.join(words)}")

    print(f"\n완료! 결과 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
