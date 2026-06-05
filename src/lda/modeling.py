"""
LDA 토픽 모델링 - 최적 K값 및 Alpha 탐색 포함
=================================================
전처리_정리 폴더의 전처리된 텍스트 파일을 읽어
LDA 토픽 모델링을 수행합니다.

지원 파일 형식:
  - CSV  : 토큰이 저장된 컬럼을 자동 감지 (공백 구분 or 리스트 형식)
  - TXT  : 한 줄 = 한 문서 (공백 구분 토큰)
  - XLSX : CSV와 동일

설치 패키지:
  pip install gensim pyLDAvis matplotlib seaborn pandas openpyxl tqdm
"""

import os
import re
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 서버/비대화형 환경에서도 동작
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 utils 임포트 가능하게 함
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.data_io import parse_token_cell, get_best_text_column
from src.utils.viz_utils import set_korean_font, save_fig

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 설정값 (필요에 따라 수정)
# ─────────────────────────────────────────────
DATA_DIR    = os.path.join("data", "processed")
OUTPUT_DIR  = os.path.join("results", "modeling_results", "LDA")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# K 탐색 범위
K_START  = 2
K_END    = 15   # 포함
K_STEP   = 1

# Alpha 후보 (최적 K 결정 후 세밀 탐색)
ALPHA_CANDIDATES = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, "symmetric", "asymmetric"]

# LDA 공통 하이퍼파라미터
ETA          = "auto"   # beta (단어-토픽 분포 하이퍼파라미터)
PASSES       = 20       # 전체 코퍼스 반복 횟수
ITERATIONS   = 400
RANDOM_STATE = 42
WORKERS      = 2        # LdaMulticore 사용 시 (None → LdaModel 단일 스레드)

# 한글 폰트 설정
set_korean_font()


# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
def load_documents(data_dir: str) -> list:
    """
    data_dir 안의 모든 csv/txt/xlsx 파일을 읽어
    [[token, token, ...], ...] 형태의 문서 리스트 반환
    """
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
            df  = pd.read_csv(fpath, encoding="utf-8-sig")
            col = get_best_text_column(df)
            docs = [parse_token_cell(v) for v in df[col]]

        else:  # xlsx / xls
            df  = pd.read_excel(fpath)
            col = get_best_text_column(df)
            docs = [parse_token_cell(v) for v in df[col]]

        docs = [d for d in docs if len(d) >= 2]
        print(f"     문서 수: {len(docs)}")
        all_docs.extend(docs)

    print(f"\n총 문서 수: {len(all_docs)}\n")
    return all_docs


# ─────────────────────────────────────────────
# 2. 사전·코퍼스 구축
# ─────────────────────────────────────────────
def build_corpus(docs: list):
    """gensim Dictionary + BoW 코퍼스 생성"""
    dictionary = corpora.Dictionary(docs)
    # 매우 희귀하거나 너무 빈번한 단어 제거
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print(f"어휘 크기: {len(dictionary)}, 문서 수: {len(corpus)}")
    return dictionary, corpus


# ─────────────────────────────────────────────
# 3. 최적 K 탐색 (Coherence c_v)
# ─────────────────────────────────────────────
def search_optimal_k(docs, dictionary, corpus,
                     k_start=K_START, k_end=K_END, k_step=K_STEP,
                     alpha="symmetric", eta=ETA):
    """
    K를 k_start ~ k_end 범위로 변화시키며 Coherence(c_v) 측정.
    반환: (k_list, coherence_list, best_k)
    """
    k_list      = list(range(k_start, k_end + 1, k_step))
    coherences  = []

    print(f"\n[ K 탐색 ] 범위: {k_start} ~ {k_end}, alpha={alpha}")
    for k in tqdm(k_list, desc="K 탐색"):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            alpha=alpha,
            eta=eta,
            passes=PASSES,
            iterations=ITERATIONS,
            random_state=RANDOM_STATE,
            per_word_topics=False,
        )
        cm = CoherenceModel(model=model, texts=docs,
                            dictionary=dictionary, coherence="c_v")
        coherences.append(cm.get_coherence())

    best_idx = int(np.argmax(coherences))
    best_k   = k_list[best_idx]
    print(f"\n최적 K = {best_k}  (Coherence = {coherences[best_idx]:.4f})")

    # ── 시각화 저장
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_list, coherences, marker="o", color="steelblue")
    ax.axvline(best_k, color="tomato", linestyle="--", label=f"Best K={best_k}")
    ax.set_xlabel("토픽 수 (K)", fontsize=12)
    ax.set_ylabel("Coherence (c_v)", fontsize=12)
    ax.set_title("LDA: K별 Coherence 점수", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "LDA_coherence_by_K.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {save_path}")

    return k_list, coherences, best_k


# ─────────────────────────────────────────────
# 4. 최적 Alpha 탐색 (고정 K 기준)
# ─────────────────────────────────────────────
def search_optimal_alpha(docs, dictionary, corpus, best_k,
                         alpha_candidates=ALPHA_CANDIDATES, eta=ETA):
    """
    alpha 후보들을 순회하며 Coherence(c_v) 측정.
    반환: (best_alpha, results_dict)
    """
    print(f"\n[ Alpha 탐색 ] K={best_k}, 후보: {alpha_candidates}")
    results = {}

    for alpha in tqdm(alpha_candidates, desc="Alpha 탐색"):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=best_k,
            alpha=alpha,
            eta=eta,
            passes=PASSES,
            iterations=ITERATIONS,
            random_state=RANDOM_STATE,
        )
        cm = CoherenceModel(model=model, texts=docs,
                            dictionary=dictionary, coherence="c_v")
        results[str(alpha)] = cm.get_coherence()

    best_alpha = max(results, key=results.get)
    print(f"\nAlpha별 Coherence:")
    for a, v in results.items():
        marker = " ← 최적" if a == best_alpha else ""
        print(f"  alpha={a:12s}  coherence={v:.4f}{marker}")

    # ── 시각화 저장 (수치형 alpha만)
    numeric_items = {k: v for k, v in results.items()
                     if k not in ("symmetric", "asymmetric")}
    if numeric_items:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = [float(k) for k in numeric_items]
        ys = list(numeric_items.values())
        ax.plot(xs, ys, marker="o", color="darkorange")
        ax.set_xlabel("Alpha 값", fontsize=12)
        ax.set_ylabel("Coherence (c_v)", fontsize=12)
        ax.set_title(f"LDA: Alpha별 Coherence (K={best_k})", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "LDA_coherence_by_alpha.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  → 저장: {save_path}")

    return best_alpha, results


# ─────────────────────────────────────────────
# 5. 최종 모델 학습 및 결과 저장
# ─────────────────────────────────────────────
def train_final_model(docs, dictionary, corpus, best_k, best_alpha, eta=ETA):
    """최적 K·alpha로 최종 LDA 모델 학습"""
    print(f"\n[ 최종 모델 학습 ] K={best_k}, alpha={best_alpha}")
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        alpha=best_alpha,
        eta=eta,
        passes=PASSES,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
        per_word_topics=True,
    )
    return model


def save_topic_words(model, best_k, topn=20):
    """토픽별 상위 단어를 CSV로 저장"""
    rows = []
    for k in range(best_k):
        words = model.show_topic(k, topn=topn)
        for rank, (word, prob) in enumerate(words, 1):
            rows.append({"topic": k + 1, "rank": rank,
                         "word": word, "probability": round(prob, 6)})
    df = pd.DataFrame(rows)
    save_path = os.path.join(OUTPUT_DIR, "LDA_topic_words.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {save_path}")
    return df


def save_document_topics(corpus, model, best_k):
    """각 문서의 토픽 분포를 CSV로 저장"""
    rows = []
    for doc_id, bow in enumerate(corpus):
        dist = dict(model.get_document_topics(bow, minimum_probability=0))
        row  = {"doc_id": doc_id}
        for k in range(best_k):
            row[f"topic_{k+1}"] = round(dist.get(k, 0.0), 6)
        row["dominant_topic"] = max(dist, key=dist.get) + 1 if dist else -1
        rows.append(row)
    df = pd.DataFrame(rows)
    save_path = os.path.join(OUTPUT_DIR, "LDA_document_topics.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {save_path}")
    return df


def plot_topic_word_heatmap(model, best_k, topn=10):
    """토픽-단어 확률 히트맵 저장"""
    data, words_all = {}, []
    for k in range(best_k):
        words = model.show_topic(k, topn=topn)
        data[f"T{k+1}"] = {w: p for w, p in words}
        words_all.extend([w for w, _ in words])
    words_all = list(dict.fromkeys(words_all))  # 중복 제거 (순서 유지)

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
    ax.set_title(f"LDA 토픽-단어 히트맵 (K={best_k})", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "LDA_topic_word_heatmap.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {save_path}")


def save_pyldavis(model, corpus, dictionary):
    """pyLDAvis 인터랙티브 HTML 저장"""
    try:
        vis = gensimvis.prepare(model, corpus, dictionary, mds="mmds")
        save_path = os.path.join(OUTPUT_DIR, "LDA_pyLDAvis.html")
        pyLDAvis.save_html(vis, save_path)
        print(f"  → 저장: {save_path}")
    except Exception as e:
        print(f"  [경고] pyLDAvis 저장 실패: {e}")


def plot_dominant_topic_dist(doc_topic_df, best_k):
    """문서별 dominant topic 분포 막대 그래프 저장"""
    counts = doc_topic_df["dominant_topic"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(max(6, best_k * 0.8), 4))
    ax.bar([f"T{t}" for t in counts.index], counts.values, color="steelblue")
    ax.set_xlabel("토픽", fontsize=12)
    ax.set_ylabel("문서 수", fontsize=12)
    ax.set_title("토픽별 문서 수 분포", fontsize=13)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "LDA_dominant_topic_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {save_path}")


# ─────────────────────────────────────────────
# 6. 메인 파이프라인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LDA 토픽 모델링 파이프라인")
    print("=" * 60)

    # ── 1. 데이터 로드
    docs = load_documents(DATA_DIR)

    # ── 2. 사전·코퍼스 구축
    dictionary, corpus = build_corpus(docs)

    # ── 3. 최적 K 탐색
    k_list, coherences, best_k = search_optimal_k(
        docs, dictionary, corpus,
        k_start=K_START, k_end=K_END, k_step=K_STEP,
    )

    # ── 4. 최적 Alpha 탐색 (고정 K 기준)
    best_alpha, alpha_results = search_optimal_alpha(
        docs, dictionary, corpus, best_k,
        alpha_candidates=ALPHA_CANDIDATES,
    )

    # ── 5. 최종 모델 학습
    model = train_final_model(docs, dictionary, corpus, best_k, best_alpha)

    # ── 6. 결과 저장
    print("\n[ 결과 저장 ]")
    topic_word_df  = save_topic_words(model, best_k, topn=20)
    doc_topic_df   = save_document_topics(corpus, model, best_k)
    plot_topic_word_heatmap(model, best_k, topn=10)
    plot_dominant_topic_dist(doc_topic_df, best_k)
    save_pyldavis(model, corpus, dictionary)

    # 모델 저장
    model_path = os.path.join(OUTPUT_DIR, f"lda_k{best_k}")
    model.save(model_path)
    print(f"  → 모델 저장: {model_path}")

    # ── 7. 토픽별 상위 단어 콘솔 출력
    print("\n[ 최종 토픽 요약 ]")
    for k in range(best_k):
        words = [w for w, _ in model.show_topic(k, topn=10)]
        print(f"  Topic {k+1:2d}: {', '.join(words)}")

    print(f"\n완료! 결과 폴더: {OUTPUT_DIR}")

    # ── 8. 탐색 결과 요약 CSV
    summary = pd.DataFrame({
        "K":          k_list,
        "coherence":  coherences,
        "best_k":     [best_k] * len(k_list),
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "LDA_K_search_summary.csv"),
                   index=False, encoding="utf-8-sig")

    alpha_df = pd.DataFrame(list(alpha_results.items()),
                            columns=["alpha", "coherence"])
    alpha_df["best_alpha"] = best_alpha
    alpha_df.to_csv(os.path.join(OUTPUT_DIR, "LDA_alpha_search_summary.csv"),
                    index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
