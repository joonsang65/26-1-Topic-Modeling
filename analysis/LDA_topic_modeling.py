"""
LDA 토픽 모델링 - Optuna 기반 최적 K & Alpha 탐색
=================================================
전처리_정리 폴더의 전처리된 텍스트 파일을 읽어
LDA 토픽 모델링을 수행합니다.

지원 파일 형식:
  - CSV  : 토큰이 저장된 컬럼을 자동 감지 (공백 구분 or 리스트 형식)
  - TXT  : 한 줄 = 한 문서 (공백 구분 토큰)
  - XLSX : CSV와 동일

설치 패키지:
  pip install gensim pyLDAvis matplotlib seaborn pandas openpyxl tqdm optuna
"""

import os
import re
import ast
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 서버/비대화형 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm import tqdm

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 설정값 (필요에 따라 수정)
# ─────────────────────────────────────────────
DATA_FILE   = r"C:\Users\SAMSUNG\Desktop\26-1\텍스트분석\텍스트 분석\최종정제_v2.csv"
TEXT_COL    = "final_text"   # 최종정제_v2.csv 의 텍스트 컬럼명
OUTPUT_DIR  = r"C:\Users\SAMSUNG\Desktop\26-1\텍스트분석\텍스트 분석\LDA_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# K 탐색 범위
K_START  = 2
K_END    = 15   # 포함

# Optuna 시도 횟수 (K × alpha 공간을 TPE 샘플러로 탐색)
N_TRIALS_LDA = 50

# Alpha 후보 (Optuna 탐색 공간으로 사용)
ALPHA_CANDIDATES = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, "symmetric", "asymmetric"]

# LDA 공통 하이퍼파라미터
ETA          = "auto"   # beta (단어-토픽 분포 하이퍼파라미터)
PASSES       = 20       # 전체 코퍼스 반복 횟수
ITERATIONS   = 400
RANDOM_STATE = 42
WORKERS      = 2        # LdaMulticore 사용 시 (None → LdaModel 단일 스레드)

# 한글 폰트 설정 (윈도우 기본)
try:
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = font_prop.get_name()
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
def _parse_token_cell(cell) -> list:
    """셀 값을 토큰 리스트로 변환 (문자열·리스트 양쪽 처리)"""
    if isinstance(cell, list):
        return [str(t).strip() for t in cell if str(t).strip()]
    s = str(cell).strip()
    if not s or s in ("nan", "None", ""):
        return []
    # 파이썬 리스트 표현식 ["a","b",...] 이면 ast 파싱
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except Exception:
            pass
    # 그 외: 공백 구분 토큰
    return s.split()


def _best_text_column(df: pd.DataFrame) -> str:
    """토큰화된 텍스트가 담긴 컬럼을 자동 선택"""
    candidates = []
    for col in df.columns:
        sample = df[col].dropna().head(5)
        lengths = sample.apply(lambda x: len(_parse_token_cell(x)))
        if lengths.mean() >= 2:
            candidates.append((col, lengths.mean()))
    if not candidates:
        raise ValueError("토큰화된 텍스트 컬럼을 찾지 못했습니다. 컬럼 이름을 직접 지정해 주세요.")
    # 평균 토큰 수가 가장 많은 컬럼 선택
    candidates.sort(key=lambda x: -x[1])
    print(f"  → 텍스트 컬럼 자동 선택: '{candidates[0][0]}'")
    return candidates[0][0]


def load_documents(data_file: str, text_col: str = TEXT_COL) -> list:
    """
    최종정제_v2.csv 를 읽어 [[token, ...], ...] 형태로 반환.
    final_text 컬럼은 공백 구분 단어열로 저장되어 있음.
    """
    print(f"[로드] {data_file}")
    df = pd.read_csv(data_file, encoding="utf-8-sig")
    if text_col not in df.columns:
        raise ValueError(f"컬럼 '{text_col}' 를 찾을 수 없습니다. 존재하는 컬럼: {df.columns.tolist()}")

    docs = [_parse_token_cell(v) for v in df[text_col]]
    docs = [d for d in docs if len(d) >= 2]
    print(f"     유효 문서 수: {len(docs)}\n")
    return docs


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
# 3. Optuna 기반 최적 K & Alpha 동시 탐색
# ─────────────────────────────────────────────
def optuna_search_lda(docs, dictionary, corpus,
                      k_start=K_START, k_end=K_END,
                      alpha_candidates=ALPHA_CANDIDATES,
                      eta=ETA, n_trials=N_TRIALS_LDA):
    """
    Optuna TPE 샘플러로 K & alpha 를 동시에 최적화.
    - Grid Search 대비: 연속 공간을 베이즈 최적화로 탐색하여
      같은 시도 횟수에서 더 좋은 하이퍼파라미터를 빠르게 발견.
    반환: (best_k, best_alpha, study)
    """
    alpha_str_list = [str(a) for a in alpha_candidates]

    def objective(trial):
        k     = trial.suggest_int("num_topics", k_start, k_end)
        a_str = trial.suggest_categorical("alpha", alpha_str_list)
        try:
            alpha = float(a_str)
        except ValueError:
            alpha = a_str  # "symmetric" / "asymmetric"

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
        return cm.get_coherence()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"\n[ Optuna LDA 탐색 ] K: {k_start}~{k_end}, "
          f"alpha 후보: {alpha_candidates}")
    print(f"  총 시도 횟수: {n_trials}  (TPE 샘플러 - 베이즈 최적화)")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_k         = study.best_params["num_topics"]
    best_alpha_str = study.best_params["alpha"]
    try:
        best_alpha = float(best_alpha_str)
    except ValueError:
        best_alpha = best_alpha_str

    print(f"\n최적 K = {best_k}, alpha = {best_alpha}")
    print(f"최고 Coherence (c_v) = {study.best_value:.4f}")

    # ── 완료된 시도 수집
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]
    ks     = [t.params["num_topics"] for t in trials]
    covs   = [t.value               for t in trials]
    alphas = [t.params["alpha"]      for t in trials]

    # ── 시각화: K vs Coherence 산점도 / Alpha vs 평균 Coherence
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # K vs Coherence (전체 시도 산점도 + K별 평균선)
    k_unique  = sorted(set(ks))
    avg_cohs  = [np.mean([covs[i] for i, kk in enumerate(ks) if kk == kv])
                 for kv in k_unique]
    axes[0].scatter(ks, covs, alpha=0.4, color="steelblue", label="각 시도")
    axes[0].plot(k_unique, avg_cohs, marker="o", color="navy",
                 linewidth=1.5, label="K별 평균")
    axes[0].axvline(best_k, color="tomato", linestyle="--",
                    label=f"Best K={best_k}")
    axes[0].set_xlabel("토픽 수 (K)", fontsize=11)
    axes[0].set_ylabel("Coherence (c_v)", fontsize=11)
    axes[0].set_title("LDA Optuna: K별 Coherence", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Alpha vs 평균 Coherence (막대)
    alpha_cov_map = {}
    for a, c in zip(alphas, covs):
        alpha_cov_map.setdefault(a, []).append(c)
    alpha_means = {a: np.mean(v) for a, v in alpha_cov_map.items()}
    bar_labels  = list(alpha_means.keys())
    bar_vals    = list(alpha_means.values())
    bar_colors  = ["tomato" if a == best_alpha_str else "darkorange"
                   for a in bar_labels]
    axes[1].bar(bar_labels, bar_vals, color=bar_colors, alpha=0.85)
    axes[1].set_xlabel("Alpha", fontsize=11)
    axes[1].set_ylabel("평균 Coherence (c_v)", fontsize=11)
    axes[1].set_title("LDA Optuna: Alpha별 평균 Coherence", fontsize=12)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "LDA_optuna_search.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {save_path}")

    # ── 시도 전체 결과 CSV
    trials_df = pd.DataFrame([{
        "trial":      t.number,
        "num_topics": t.params.get("num_topics"),
        "alpha":      t.params.get("alpha"),
        "coherence":  t.value,
    } for t in trials])
    csv_path = os.path.join(OUTPUT_DIR, "LDA_optuna_trials.csv")
    trials_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {csv_path}")

    # ── K별 요약 출력
    print(f"\nK별 평균 Coherence:")
    for kv, ac in zip(k_unique, avg_cohs):
        marker = " ← 최적" if kv == best_k else ""
        print(f"  K={kv:2d}  평균 Coherence={ac:.4f}{marker}")

    return best_k, best_alpha, study


# ─────────────────────────────────────────────
# 4. 최종 모델 학습 및 결과 저장
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
# 5. 메인 파이프라인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LDA 토픽 모델링 파이프라인 (Optuna 최적화)")
    print("=" * 60)

    # ── 1. 데이터 로드
    docs = load_documents(DATA_FILE, TEXT_COL)

    # ── 2. 사전·코퍼스 구축
    dictionary, corpus = build_corpus(docs)

    # ── 3. Optuna 기반 최적 K & Alpha 동시 탐색
    #    (기존 grid search 대비 TPE 베이즈 최적화로 더 세밀한 탐색)
    best_k, best_alpha, study = optuna_search_lda(
        docs, dictionary, corpus,
        k_start=K_START, k_end=K_END,
        alpha_candidates=ALPHA_CANDIDATES,
        n_trials=N_TRIALS_LDA,
    )

    # ── 4. 최종 모델 학습
    model = train_final_model(docs, dictionary, corpus, best_k, best_alpha)

    # ── 5. 결과 저장
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

    # ── 6. 토픽별 상위 단어 콘솔 출력
    print("\n[ 최종 토픽 요약 ]")
    for k in range(best_k):
        words = [w for w, _ in model.show_topic(k, topn=10)]
        print(f"  Topic {k+1:2d}: {', '.join(words)}")

    print(f"\n완료! 결과 폴더: {OUTPUT_DIR}")

    # ── 7. Optuna 탐색 요약 CSV (K별 평균 coherence)
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]
    summary_df = pd.DataFrame([{
        "trial":      t.number,
        "num_topics": t.params.get("num_topics"),
        "alpha":      t.params.get("alpha"),
        "coherence":  t.value,
        "best_k":     best_k,
        "best_alpha": str(best_alpha),
    } for t in trials])
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "LDA_optuna_summary.csv"),
        index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
