"""
Extract the top sentences for each BTM topic from saved topic-word probabilities.

Input
- src/btm/results/BTM_topic_words.csv
  Long-format topic-word probabilities with columns: topic, rank, word, prob
- data/processed/토크나이징_전_전처리.csv
  Source sentence/comment data. The text column is used as the output sentence.

Output
- src/btm/results/BTM_topic_sentences.csv
  Top 30 sentences for each topic.
- src/btm/results/BTM_sentence_topic_probabilities.csv
  Topic scores/probabilities for every sentence.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

TOPIC_WORDS_FILE = os.path.join(OUTPUT_DIR, "BTM_topic_words.csv")
SENTENCE_FILE = os.path.join(DATA_DIR, "토크나이징_전_전처리.csv")

TOP_N_SENTENCES = 50
TEXT_COLUMN = "text"
SCORING_TEXT_FALLBACK_COLUMN = "cleaned_text"


def load_topic_words(path: str = TOPIC_WORDS_FILE) -> tuple[list[int], dict[int, list[tuple[str, float]]]]:
    """Load topic-word probabilities grouped by topic."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required_columns = {"topic", "word", "prob"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df = df.dropna(subset=["topic", "word", "prob"]).copy()
    df["topic"] = df["topic"].astype(int)
    df["word"] = df["word"].astype(str).str.strip()
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df[(df["word"] != "") & df["prob"].notna() & (df["prob"] > 0)]

    topic_words: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for topic, group in df.sort_values(["topic", "prob"], ascending=[True, False]).groupby("topic"):
        topic_words[int(topic)] = list(zip(group["word"], group["prob"].astype(float)))

    topics = sorted(topic_words)
    if not topics:
        raise ValueError(f"No valid topic-word probabilities found in {path}")
    return topics, topic_words


def load_sentences(path: str = SENTENCE_FILE) -> pd.DataFrame:
    """Load source text rows and keep only non-empty sentences."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"'{TEXT_COLUMN}' column not found in {path}")

    df = df.reset_index().rename(columns={"index": "source_row"})
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)
    return df


def normalize_text(value: object) -> str:
    """Normalize text for simple Korean/English substring matching."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def count_occurrences(text: str, word: str) -> int:
    """Count word appearances in a sentence.

    Korean topic words often do not have reliable whitespace boundaries, so a
    substring count is more useful here than a word-boundary regex.
    """
    normalized_word = normalize_text(word)
    if not normalized_word:
        return 0
    return text.count(normalized_word)


def score_sentence(text: str, topic_terms: list[tuple[str, float]]) -> tuple[float, int, str]:
    """Return weighted topic score, matched term count, and matched word list."""
    raw_score = 0.0
    matched_count = 0
    matched_words: list[str] = []

    for word, prob in topic_terms:
        count = count_occurrences(text, word)
        if count <= 0:
            continue
        raw_score += prob * count
        matched_count += count
        matched_words.append(word)

    # Slightly reduce the advantage of very long comments while preserving
    # multiple meaningful matches.
    length_penalty = math.sqrt(max(len(text), 1))
    score = raw_score / length_penalty
    return score, matched_count, ", ".join(matched_words)


def build_sentence_topic_scores(
    sentences_df: pd.DataFrame,
    topics: list[int],
    topic_words: dict[int, list[tuple[str, float]]],
) -> pd.DataFrame:
    """Calculate topic scores and per-sentence normalized topic probabilities."""
    rows = []
    scoring_column = (
        SCORING_TEXT_FALLBACK_COLUMN
        if SCORING_TEXT_FALLBACK_COLUMN in sentences_df.columns
        else TEXT_COLUMN
    )

    for doc_id, row in sentences_df.iterrows():
        scoring_text = normalize_text(row[scoring_column])
        output_row = {
            "doc_id": int(doc_id),
            "source_row": int(row["source_row"]),
            "text": row[TEXT_COLUMN],
        }
        if "cid" in sentences_df.columns:
            output_row["cid"] = row["cid"]

        topic_raw_scores = {}
        for topic in topics:
            score, matched_count, matched_words = score_sentence(scoring_text, topic_words[topic])
            topic_raw_scores[topic] = score
            output_row[f"topic_{topic}_score"] = float(score)
            output_row[f"topic_{topic}_matched_count"] = int(matched_count)
            output_row[f"topic_{topic}_matched_words"] = matched_words

        total_score = float(sum(topic_raw_scores.values()))
        for topic in topics:
            prob = topic_raw_scores[topic] / total_score if total_score > 0 else 0.0
            output_row[f"topic_{topic}_prob"] = float(prob)

        if total_score > 0:
            dominant_topic = max(topics, key=lambda t: topic_raw_scores[t])
            dominant_prob = output_row[f"topic_{dominant_topic}_prob"]
        else:
            dominant_topic = np.nan
            dominant_prob = 0.0

        output_row["dominant_topic"] = dominant_topic
        output_row["dominant_topic_prob"] = float(dominant_prob)
        rows.append(output_row)

    return pd.DataFrame(rows)


def save_topic_sentences(sentence_prob_df: pd.DataFrame, topics: list[int]) -> tuple[str, str]:
    """Save all sentence probabilities and the top N sentences for each topic."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_sentence_probs_path = os.path.join(OUTPUT_DIR, "BTM_sentence_topic_probabilities.csv")
    sentence_prob_df.to_csv(all_sentence_probs_path, index=False, encoding="utf-8-sig")

    top_sentence_rows = []
    for topic in topics:
        prob_col = f"topic_{topic}_prob"
        score_col = f"topic_{topic}_score"
        matched_count_col = f"topic_{topic}_matched_count"
        matched_words_col = f"topic_{topic}_matched_words"

        top_df = (
            sentence_prob_df[sentence_prob_df[score_col] > 0]
            .sort_values([prob_col, score_col, matched_count_col], ascending=[False, False, False])
            .head(TOP_N_SENTENCES)
        )

        for rank, (_, row) in enumerate(top_df.iterrows(), 1):
            result_row = {
                "topic": int(topic),
                "rank": int(rank),
                "prob": float(row[prob_col]),
                "score": float(row[score_col]),
                "matched_count": int(row[matched_count_col]),
                "matched_words": row[matched_words_col],
                "doc_id": int(row["doc_id"]),
                "source_row": int(row["source_row"]),
                "text": row["text"],
            }
            if "cid" in sentence_prob_df.columns:
                result_row["cid"] = row["cid"]
            top_sentence_rows.append(result_row)

    top_sentences_path = os.path.join(OUTPUT_DIR, "BTM_topic_sentences.csv")
    pd.DataFrame(top_sentence_rows).to_csv(top_sentences_path, index=False, encoding="utf-8-sig")

    return top_sentences_path, all_sentence_probs_path


def main() -> None:
    topics, topic_words = load_topic_words()
    sentences_df = load_sentences()
    sentence_prob_df = build_sentence_topic_scores(sentences_df, topics, topic_words)
    top_sentences_path, all_sentence_probs_path = save_topic_sentences(sentence_prob_df, topics)

    print(f"Topic words: {TOPIC_WORDS_FILE}")
    print(f"Source sentences: {SENTENCE_FILE}")
    print(f"Top {TOP_N_SENTENCES} sentences per topic: {top_sentences_path}")
    print(f"All sentence topic probabilities: {all_sentence_probs_path}")


if __name__ == "__main__":
    main()
