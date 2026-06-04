"""
BTM 토픽 모델링 - Optuna 기반 최적 K·Alpha·Beta 탐색
=================================================
Biterm Topic Model (BTM) 은 문서가 짧을 때 효과적인 토픽 모델입니다.
(뉴스 헤드라인, SNS 게시물, 리뷰 등 단문 텍스트에 특히 유리)

전처리_정리 폴더의 전처리된 텍스트 파일을 읽어 BTM을 수행합니다.

설치 패키지:
  pip install bitermplus matplotlib seaborn pandas openpyxl tqdm numpy scipy optuna

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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 설정값 (필요에 따라 수정)
# ─────────────────────────────────────────────
DATA_FILE   = r"C:\Users\SAMSUNG\Desktop\26-1\텍스트분석\텍스트 분석\최종정제_v2.csv"
TEXT_COL    = "final_text"   # 최종정제_v2.csv 의 텍스트 컬럼명
OUTPUT_DIR  = r"C:\Users\SAMSUNG\Desktop\26-1\텍스트분석\텍스트 분석\BTM_결과"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# K 탐색 범위
K_START     = 2
K_END       = 15

# Optuna 시도 횟수 (K × alpha × beta 공간을 TPE 샘플러로 탐색)
N_TRIALS_BTM = 40

# BTM 하이퍼파라미터 탐색 범위
ALPHA_LOW   = 0.01     # alpha 탐색 하한 (디리클레 파라미터)
ALPHA_HIGH  = 5.0      # alpha 탐색 상한
BETA_LOW    = 0.001    # beta 탐색 하한
BETA_HIGH   = 0.5      # beta 탐색 상한
N_ITER      = 200      # 깁스 샘플링 반복 수

# 복합 점수 가중치 (Coherence ↑ + Perplexity ↓)
WEIGHT_COH  = 0.5      # Coherence 가중치
WEIGHT_PPL  = 0.5      # Perplexity(역방향) 가중치

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
              alpha: float, beta: float, n_iter=N_ITER):
    """
    bitermplus 로 BTM 학습.
    반환: (btm_model, docs_idx)
    """
    import bitermplus as btm

    docs_idx = [[vocab[w] for w in doc if w in vocab]
                for doc in docs_filtered]

    vocab_list = [None] * len(vocab)
    for w, i in vocab.items():
        vocab_list[i] = w

    model = btm.BTM(
        X          = docs_idx,
        vocabulary = vocab_list,
        T          = num_topics,
        M          = 20,
        alpha      = alpha,
        beta       = beta,
    )
    model.fit(docs_idx, iterations=n_iter)
    return model, docs_idx


# ─────────────────────────────────────────────
# 4. 토픽 코히런스 계산 (BTM 내부용)
# ─────────────────────────────────────────────
def _topic_coherence_btm(model, docs_idx: list, vocab: dict,
                         topn: int = 10) -> float:
    """
    BTM 모델의 토픽 코히런스 계산 (UCI Pointwise MI 근사).
    공식: Coherence = mean over topics of mean PMI(w_i, w_j)
    """
    phi = model.matrix_words_topics_  # shape: (vocab, K)

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
                c_ij   = cooc_freq.get(pair, 0) + 1
                c_i    = word_freq.get(wi, 0)  + 1
                c_j    = word_freq.get(wj, 0)  + 1
                pmi    = np.log((c_ij * n_docs) / (c_i * c_j + 1e-12) + 1e-12)
                pmi_scores.append(pmi)
        coherences.append(np.mean(pmi_scores) if pmi_scores else 0.0)
    return float(np.mean(coherences))


# ─────────────────────────────────────────────
# 5. Optuna 기반 최적 K & 하이퍼파라미터 탐색
# ─────────────────────────────────────────────
def optuna_search_btm(docs_filtered: list, vocab: dict,
                      k_start=K_START, k_end=K_END,
                      alpha_low=ALPHA_LOW, alpha_high=ALPHA_HIGH,
                      beta_low=BETA_LOW,  beta_high=BETA_HIGH,
                      n_iter=N_ITER, n_trials=N_TRIALS_BTM):
    """
    Optuna TPE 샘플러로 K, alpha, beta 를 동시에 최적화.

    best_k 선택 기준:
      - Coherence (높을수록 ↑) + Perplexity (낮을수록 ↓) 복합 점수
      - 두 지표를 Min-Max 정규화 후 가중 평균:
          score = WEIGHT_COH × coh_norm + WEIGHT_PPL × (1 − ppl_norm)

    반환: (study, best_k, best_alpha, best_beta)
    """
    import bitermplus as btm_lib

    docs_idx = [[vocab[w] for w in doc if w in vocab] for doc in docs_filtered]
    vocab_list = [None] * len(vocab)
    for w, i in vocab.items():
        vocab_list[i] = w

    def objective(trial):
        k     = trial.suggest_int("num_topics", k_start, k_end)
        alpha = trial.suggest_float("alpha", alpha_low, alpha_high, log=True)
        beta  = trial.suggest_float("beta",  beta_low,  beta_high,  log=True)

        model = btm_lib.BTM(
            X=docs_idx, vocabulary=vocab_list,
            T=k, M=20, alpha=alpha, beta=beta,
        )
        model.fit(docs_idx, iterations=n_iter)

        # Perplexity (낮을수록 좋음)
        try:
            ppl = float(model.perplexity_)
        except Exception:
            ppl = float("inf")

        # Coherence (높을수록 좋음)
        coh = _topic_coherence_btm(model, docs_idx, vocab, topn=10)

        # Optuna 내부 기록용 (복합 점수는 사후 계산)
        trial.set_user_attr("perplexity", ppl)
        trial.set_user_attr("coherence",  coh)

        return coh  # Optuna 최적화 기준: Coherence 최대화

    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"\n[ Optuna BTM 탐색 ] K: {k_start}~{k_end}")
    print(f"  alpha: [{alpha_low}, {alpha_high}] (log scale)")
    print(f"  beta:  [{beta_low}, {beta_high}] (log scale)")
    print(f"  총 시도 횟수: {n_trials}  (TPE 샘플러 - 베이즈 최적화)")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── 완료된 시도 수집
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]

    ppls = np.array([t.user_attrs["perplexity"] for t in trials])
    cohs = np.array([t.user_attrs["coherence"]   for t in trials])
    ks   = np.array([t.params["num_topics"]       for t in trials])

    # ── 복합 점수 계산 (Coherence↑ + Perplexity↓)
    valid = np.isfinite(ppls) & np.isfinite(cohs)

    ppl_min, ppl_max = ppls[valid].min(), ppls[valid].max()
    coh_min, coh_max = cohs[valid].min(), cohs[valid].max()

    ppl_norm = np.where(valid,
                        (ppls - ppl_min) / (ppl_max - ppl_min + 1e-12),
                        1.0)
    coh_norm = np.where(valid,
                        (cohs - coh_min) / (coh_max - coh_min + 1e-12),
                        0.0)
    combined = WEIGHT_COH * coh_norm + WEIGHT_PPL * (1.0 - ppl_norm)

    best_rel   = int(np.argmax(combined))
    best_trial = trials[best_rel]
    best_k     = best_trial.params["num_topics"]
    best_alpha = best_trial.params["alpha"]
    best_beta  = best_trial.params["beta"]

    # ── 결과 출력
    print(f"\n[ Optuna BTM 탐색 결과 ]")
    print(f"  최적 K = {best_k},  alpha = {best_alpha:.4f},  beta = {best_beta:.4f}")
    print(f"  Coherence = {best_trial.user_attrs['coherence']:.4f}, "
          f"Perplexity = {best_trial.user_attrs['perplexity']:.2f}")
    print(f"  복합 점수 = {combined[best_rel]:.4f}  "
          f"(가중치 Coh:{WEIGHT_COH} / Ppl:{WEIGHT_PPL})")

    # K별 평균 지표 출력
    print(f"\nK별 평균 지표 (Coherence ↑ + Perplexity ↓ 복합 점수):")
    k_unique = sorted(set(ks))
    for kv in k_unique:
        mask     = ks == kv
        avg_coh  = np.nanmean(cohs[mask])
        avg_ppl  = np.nanmean(ppls[mask])
        avg_comb = np.nanmean(combined[mask])
        marker   = " ← 최적(복합 점수)" if kv == best_k else ""
        print(f"  K={kv:2d}  Coherence={avg_coh:.4f}  "
              f"Perplexity={avg_ppl:8.2f}  복합={avg_comb:.4f}{marker}")

    # ── 시각화 (3개 패널: Perplexity / Coherence / 복합 점수)
    avg_ppls  = [np.nanmean(ppls[ks == kv])     for kv in k_unique]
    avg_cohs  = [np.nanmean(cohs[ks == kv])     for kv in k_unique]
    avg_combs = [np.nanmean(combined[ks == kv]) for kv in k_unique]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(k_unique, avg_ppls, marker="o", color="tomato")
    axes[0].axvline(best_k, color="steelblue", linestyle="--",
                    label=f"Best K={best_k}")
    axes[0].set_xlabel("토픽 수 (K)", fontsize=11)
    axes[0].set_ylabel("평균 Perplexity", fontsize=11)
    axes[0].set_title("BTM Optuna: K별 Perplexity", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_unique, avg_cohs, marker="o", color="steelblue")
    axes[1].axvline(best_k, color="tomato", linestyle="--",
                    label=f"Best K={best_k}")
    axes[1].set_xlabel("토픽 수 (K)", fontsize=11)
    axes[1].set_ylabel("평균 Coherence (PMI)", fontsize=11)
    axes[1].set_title("BTM Optuna: K별 Coherence", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_unique, avg_combs, marker="o", color="mediumpurple")
    axes[2].axvline(best_k, color="tomato", linestyle="--",
                    label=f"Best K={best_k} (복합)")
    axes[2].set_xlabel("토픽 수 (K)", fontsize=11)
    axes[2].set_ylabel("복합 점수", fontsize=11)
    axes[2].set_title(
        f"BTM Optuna: K별 복합 점수\n"
        f"(Coh↑ × {WEIGHT_COH} + (1-Ppl↓) × {WEIGHT_PPL})", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "BTM_optuna_search.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  → 저장: {save_path}")

    # ── 시도 전체 결과 CSV
    trials_df = pd.DataFrame([{
        "trial":        t.number,
        "num_topics":   t.params["num_topics"],
        "alpha":        t.params["alpha"],
        "beta":         t.params["beta"],
        "coherence":    t.user_attrs["coherence"],
        "perplexity":   t.user_attrs["perplexity"],
        "combined_score": combined[i],
    } for i, t in enumerate(trials)])
    csv_path = os.path.join(OUTPUT_DIR, "BTM_optuna_trials.csv")
    trials_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  → 저장: {csv_path}")

    return study, best_k, best_alpha, best_beta


# ─────────────────────────────────────────────
# 6. 결과 저장
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
    phi     = model.matrix_words_topics_

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
        top_ids = np.argsort(phi[:, k])[-topn:][::-1]
        words   = [id2word[wid] for wid in top_ids]
        probs   = [float(phi[wid, k]) for wid in top_ids]
        ax = axes[k]
        ax.barh(words[::-1], probs[::-1], color="steelblue")
        ax.set_title(f"Topic {k+1}", fontsize=11)
        ax.set_xlabel("확률", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

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
# 7. 메인 파이프라인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BTM 토픽 모델링 파이프라인 (Optuna 최적화)")
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
    docs = load_documents(DATA_FILE, TEXT_COL)

    # ── 3. 어휘 구축
    vocab, docs_filtered = build_vocab(docs, min_count=2, max_ratio=0.95)

    # ── 4. Optuna 기반 최적 K, alpha, beta 탐색
    #    - Grid Search 대비: 연속 파라미터(alpha·beta)를 TPE로 세밀하게 탐색
    #    - best_k 선택: Coherence(↑) + Perplexity(↓) 복합 점수 사용
    study, best_k, best_alpha, best_beta = optuna_search_btm(
        docs_filtered, vocab,
        k_start=K_START, k_end=K_END,
        alpha_low=ALPHA_LOW, alpha_high=ALPHA_HIGH,
        beta_low=BETA_LOW,   beta_high=BETA_HIGH,
        n_iter=N_ITER, n_trials=N_TRIALS_BTM,
    )

    # ── 5. 최종 모델 학습 (최적 파라미터로)
    print(f"\n[ 최종 BTM 모델 학습 ] K={best_k}, "
          f"alpha={best_alpha:.4f}, beta={best_beta:.4f}")
    model, docs_idx = train_btm(
        docs_filtered, vocab,
        num_topics=best_k,
        alpha=best_alpha,
        beta=best_beta,
        n_iter=N_ITER,
    )

    # ── 6. 결과 저장
    print("\n[ 결과 저장 ]")
    topic_word_df = save_btm_topic_words(model, vocab, best_k, topn=20)
    doc_topic_df  = save_btm_document_topics(model, docs_idx, best_k)
    plot_btm_topic_word_heatmap(model, vocab, best_k, topn=10)
    plot_btm_topic_bar(model, vocab, best_k, topn=10)
    plot_btm_dominant_topic_dist(doc_topic_df, best_k)

    # ── 7. Optuna 탐색 요약 CSV
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]
    summary = pd.DataFrame([{
        "trial":        t.number,
        "K":            t.params["num_topics"],
        "alpha":        t.params["alpha"],
        "beta":         t.params["beta"],
        "coherence":    t.user_attrs["coherence"],
        "perplexity":   t.user_attrs["perplexity"],
        "best_k":       best_k,
        "best_alpha":   best_alpha,
        "best_beta":    best_beta,
    } for t in trials])
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "BTM_optuna_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"  → 저장: {os.path.join(OUTPUT_DIR, 'BTM_optuna_summary.csv')}")

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
