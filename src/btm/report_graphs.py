"""
Generate the requested word-frequency, BTM K-analysis, and topic-word graphs.

Input:
    data/processed/최종정제_v2.csv

Outputs:
    results/modeling_results/BTM/requested_graphs/top20_word_frequency.png
    results/modeling_results/BTM/requested_graphs/top_word_frequency_share.csv
    results/modeling_results/BTM/requested_graphs/BTM_K_metric_*.png
    results/modeling_results/BTM/requested_graphs/BTM_topic_word_distribution_K4_topic*.png
    results/modeling_results/BTM/requested_graphs/BTM_K_metrics_summary.csv
    results/modeling_results/BTM/requested_graphs/BTM_topic_words_K4.csv
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import bitermplus as btm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from scipy import sparse
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.data_io import get_best_text_column, parse_token_cell
from src.utils.viz_utils import set_korean_font


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "최종정제_v2.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "modeling_results" / "BTM" / "requested_graphs"
)
DEFAULT_PYLDAVIS_HTML = PROJECT_ROOT / "src" / "btm" / "results" / "BTM_pyLDAvis_K4.html"
RANDOM_SEED = 42


def set_plot_font() -> None:
    set_korean_font()
    windows_font = Path("C:/Windows/Fonts/malgun.ttf")
    if windows_font.exists():
        fm.fontManager.addfont(str(windows_font))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_documents(csv_path: Path) -> list[list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    text_col = get_best_text_column(df)
    docs = [parse_token_cell(value) for value in df[text_col]]
    docs = [[str(token).strip() for token in doc if str(token).strip()] for doc in docs]
    docs = [doc for doc in docs if len(doc) >= 2]

    if not docs:
        raise ValueError(f"No valid tokenized documents found in column '{text_col}'.")

    print(f"[load] file={csv_path}")
    print(f"[load] text_column={text_col}, documents={len(docs):,}")
    return docs


def prepare_btm_inputs(docs: list[list[str]]):
    dictionary = Dictionary(docs)
    vocab = {word: idx for idx, word in enumerate(sorted(dictionary.token2id))}
    vocab_list = np.array([None] * len(vocab), dtype=object)
    for word, idx in vocab.items():
        vocab_list[idx] = word

    docs_idx = [[vocab[word] for word in doc if word in vocab] for doc in docs]
    docs_idx = [doc for doc in docs_idx if len(doc) >= 2]

    rows, cols, data = [], [], []
    for row_idx, doc in enumerate(docs_idx):
        for word_id in doc:
            rows.append(row_idx)
            cols.append(word_id)
            data.append(1)

    x_matrix = sparse.csr_matrix(
        (
            np.array(data, dtype=np.int32),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(len(docs_idx), len(vocab)),
    )
    x_matrix.indices = x_matrix.indices.astype(np.int32)
    x_matrix.indptr = x_matrix.indptr.astype(np.int32)

    biterms = btm.get_biterms(docs_idx)
    return dictionary, vocab_list, docs_idx, x_matrix, biterms


def save_top_word_frequency_outputs(
    docs: list[list[str]], output_dir: Path, top_n: int
) -> tuple[Path, Path]:
    set_plot_font()
    counter = Counter(token for doc in docs for token in doc)
    total_tokens = sum(counter.values())
    rows = []

    cumulative_count = 0
    for rank, (word, count) in enumerate(counter.most_common(top_n), start=1):
        cumulative_count += count
        rows.append(
            {
                "rank": rank,
                "word": word,
                "count": count,
                "share": count / total_tokens if total_tokens else 0.0,
                "share_percent": (count / total_tokens * 100) if total_tokens else 0.0,
                "cumulative_count": cumulative_count,
                "cumulative_share": cumulative_count / total_tokens if total_tokens else 0.0,
                "cumulative_share_percent": (
                    cumulative_count / total_tokens * 100 if total_tokens else 0.0
                ),
            }
        )

    freq_df = pd.DataFrame(rows)
    csv_path = output_dir / "top_word_frequency_share.csv"
    freq_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_df = freq_df.sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 9))
    bars = ax.barh(plot_df["word"], plot_df["count"], color="seagreen", alpha=0.95)
    ax.set_title(f"상위 {top_n}개 단어 빈도", fontsize=18)
    ax.set_xlabel("빈도", fontsize=12)
    ax.set_ylabel("단어", fontsize=12)
    ax.grid(axis="x", alpha=0.35)

    max_count = int(freq_df["count"].max()) if not freq_df.empty else 0
    ax.set_xlim(0, max_count * 1.08 if max_count else 1)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max_count * 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    png_path = output_dir / f"top{top_n}_word_frequency.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def get_topic_words(model, top_n: int) -> list[list[str]]:
    phi = model.matrix_topics_words_
    vocab = model.vocabulary_

    topic_words = []
    for topic_idx in range(phi.shape[0]):
        top_ids = np.argsort(phi[topic_idx, :])[-top_n:][::-1]
        topic_words.append([str(vocab[word_id]) for word_id in top_ids])
    return topic_words


def get_coherence(
    model,
    docs: list[list[str]],
    dictionary: Dictionary,
    metric: str,
    top_n: int = 10,
) -> float:
    topic_words = get_topic_words(model, top_n=top_n)
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=docs,
        dictionary=dictionary,
        coherence=metric,
        processes=1,
    )
    return float(coherence_model.get_coherence())


def calculate_biterm_perplexity(model, biterms, eps: float = 1e-300) -> float:
    phi = model.matrix_topics_words_
    theta = model.theta_

    log_likelihood = 0.0
    n_biterms = 0
    for doc_biterms in biterms:
        for word_1, word_2 in doc_biterms:
            prob = np.sum(theta * phi[:, word_1] * phi[:, word_2])
            log_likelihood += np.log(max(float(prob), eps))
            n_biterms += 1

    if n_biterms == 0:
        return float("nan")
    return float(np.exp(-log_likelihood / n_biterms))


def train_and_score_k(
    k: int,
    docs: list[list[str]],
    dictionary: Dictionary,
    vocab_list,
    x_matrix,
    biterms,
    n_iter: int,
    top_n: int,
):
    model = btm.BTM(
        x_matrix,
        vocab_list,
        T=k,
        M=20,
        alpha=0.1,
        beta=0.01,
        seed=RANDOM_SEED,
    )
    model.fit(biterms, iterations=n_iter)

    scores = {
        "K": k,
        "C_umass": get_coherence(model, docs, dictionary, "u_mass", top_n=top_n),
        "C_v": get_coherence(model, docs, dictionary, "c_v", top_n=top_n),
        "C_npmi": get_coherence(model, docs, dictionary, "c_npmi", top_n=top_n),
        "Perplexity": calculate_biterm_perplexity(model, biterms),
    }
    scores["Log_Perplexity"] = float(np.log(scores["Perplexity"]))
    return scores, model


def save_metric_plots(metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    set_plot_font()
    plot_specs = [
        (
            "C_umass",
            "1. C_umass",
            "steelblue",
            "o",
            "BTM_K_metric_1_C_umass.png",
        ),
        (
            "C_v",
            "2. C_v",
            "seagreen",
            "s",
            "BTM_K_metric_2_C_v.png",
        ),
        (
            "C_npmi",
            "3. C_npmi",
            "darkorange",
            "^",
            "BTM_K_metric_3_C_npmi.png",
        ),
        (
            "Log_Perplexity",
            "4Log_Perplexity",
            "crimson",
            "x",
            "BTM_K_metric_4_Log_Perplexity.png",
        ),
    ]

    saved_paths = []
    for column, title, color, marker, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.lineplot(
            data=metrics_df,
            x="K",
            y=column,
            marker=marker,
            linewidth=2,
            markersize=8,
            color=color,
            ax=ax,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("토픽 개수 (K)")
        ax.set_ylabel("Log_Perplexity" if column == "Log_Perplexity" else "Coherence Score")
        ax.set_xticks(metrics_df["K"].tolist())
        ax.grid(True, alpha=0.35)

        fig.tight_layout()
        save_path = output_dir / filename
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)
    return saved_paths


def save_topic_word_outputs(
    model,
    selected_k: int,
    output_dir: Path,
    top_n: int,
    pyldavis_html: Path,
    lambda_value: float,
) -> list[Path]:
    set_plot_font()
    if not pyldavis_html.exists():
        raise FileNotFoundError(f"pyLDAvis HTML not found: {pyldavis_html}")

    html_text = pyldavis_html.read_text(encoding="utf-8")
    match = re.search(r"var\s+\w+_data\s*=\s*(\{.*?\});", html_text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find pyLDAvis JSON data in: {pyldavis_html}")

    pyldavis_data = json.loads(match.group(1))
    tinfo = pd.DataFrame(pyldavis_data["tinfo"])
    tinfo["relevance"] = (
        lambda_value * tinfo["logprob"] + (1 - lambda_value) * tinfo["loglift"]
    )

    rows = []
    for topic_idx in range(selected_k):
        topic_name = f"Topic{topic_idx + 1}"
        topic_data = (
            tinfo[tinfo["Category"] == topic_name]
            .assign(topic_share=lambda df: np.exp(df["logprob"]))
            .sort_values(["topic_share", "relevance"], ascending=[False, False])
            .head(top_n)
            .copy()
        )
        for rank, row in enumerate(topic_data.itertuples(index=False), start=1):
            rows.append(
                {
                    "topic": topic_idx + 1,
                    "rank": rank,
                    "word": row.Term,
                    "freq": float(row.Freq),
                    "total": float(row.Total),
                    "topic_share": float(row.topic_share),
                    "topic_share_percent": float(row.topic_share * 100),
                    "logprob": float(row.logprob),
                    "loglift": float(row.loglift),
                    "relevance_lambda_0_6": float(row.relevance),
                }
            )

    topic_words_df = pd.DataFrame(rows)
    topic_words_df.to_csv(
        output_dir / f"BTM_topic_words_K{selected_k}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    saved_paths = []
    for topic_idx in range(selected_k):
        fig, ax = plt.subplots(figsize=(9, 7.2))
        topic_data = topic_words_df[topic_words_df["topic"] == topic_idx + 1].copy()
        topic_data = topic_data.sort_values(["topic_share", "relevance_lambda_0_6"], ascending=[True, True])
        bars = ax.barh(topic_data["word"], topic_data["topic_share_percent"], color="steelblue")
        max_share = float(topic_data["topic_share_percent"].max()) if not topic_data.empty else 0.0
        ax.set_xlim(0, max_share * 1.12 if max_share else 1)
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + max_share * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}%",
                va="center",
                fontsize=8,
            )
        ax.set_title(
            f"Topic {topic_idx + 1} 상위 단어",
            fontsize=14,
        )
        ax.set_xlabel("토픽 내 단어 비중 (%)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.2)

        fig.tight_layout()
        save_path = (
            output_dir
            / f"BTM_topic_word_distribution_K{selected_k}_topic{topic_idx + 1}.png"
        )
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--selected-k", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--freq-top-n", type=int, default=30)
    parser.add_argument("--pyldavis-html", type=Path, default=DEFAULT_PYLDAVIS_HTML)
    parser.add_argument("--lambda-value", type=float, default=0.6)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.k_min <= args.selected_k <= args.k_max):
        raise ValueError("--selected-k must be between --k-min and --k-max.")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_plot_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(args.data)
    freq_csv, freq_png = save_top_word_frequency_outputs(
        docs, args.output_dir, top_n=args.freq_top_n
    )
    dictionary, vocab_list, _docs_idx, x_matrix, biterms = prepare_btm_inputs(docs)
    print(f"[prepare] vocab={len(vocab_list):,}, k_range={args.k_min}..{args.k_max}")

    metrics = []
    selected_model = None
    for k in tqdm(range(args.k_min, args.k_max + 1), desc="Training BTM by K"):
        scores, model = train_and_score_k(
            k=k,
            docs=docs,
            dictionary=dictionary,
            vocab_list=vocab_list,
            x_matrix=x_matrix,
            biterms=biterms,
            n_iter=args.n_iter,
            top_n=args.top_n,
        )
        metrics.append(scores)
        if k == args.selected_k:
            selected_model = model
        print(
            "[score] "
            f"K={k}, C_umass={scores['C_umass']:.4f}, C_v={scores['C_v']:.4f}, "
            f"C_npmi={scores['C_npmi']:.4f}, "
            f"Perplexity={scores['Perplexity']:.4f}, "
            f"Log_Perplexity={scores['Log_Perplexity']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_csv = args.output_dir / "BTM_K_metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    metric_pngs = save_metric_plots(metrics_df, args.output_dir)
    topic_pngs = save_topic_word_outputs(
        selected_model,
        selected_k=args.selected_k,
        output_dir=args.output_dir,
        top_n=args.top_n,
        pyldavis_html=args.pyldavis_html,
        lambda_value=args.lambda_value,
    )

    best_k = int(metrics_df.loc[metrics_df["C_umass"].idxmax(), "K"])
    print("[done] saved files:")
    print(f"  word_frequency_csv: {freq_csv}")
    print(f"  word_frequency_plot: {freq_png}")
    print(f"  metrics_csv: {metrics_csv}")
    for path in metric_pngs:
        print(f"  metric_plot: {path}")
    for path in topic_pngs:
        print(f"  topic_plot: {path}")
    print(f"[done] best K by C_umass: {best_k}")


if __name__ == "__main__":
    main()
