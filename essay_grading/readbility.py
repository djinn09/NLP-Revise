
import numpy as np
import textstat
from numpy.linalg import norm


def get_readability_metrics(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "gunning_fog": textstat.gunning_fog(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "difficult_words": textstat.difficult_words(text),
        "sentence_count": textstat.sentence_count(text),
        "avg_sentence_length": textstat.avg_sentence_length(text),
        "syllable_count": textstat.syllable_count(text),
        "lexicon_count": textstat.lexicon_count(text, removepunct=True),
        "text_standard": textstat.text_standard(text, float_output=True),
    }


def compare_metrics(metrics1, metrics2):
    return {k: round(abs(metrics1[k] - metrics2[k]), 4) for k in sorted(metrics1)}


def cosine_similarity(m1, m2):
    v1 = np.array(list(m1.values()))
    v2 = np.array(list(m2.values()))
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


if __name__ == "__main__":
    # Example texts
    text1 = "Education is the passport to the future, for tomorrow belongs to those who prepare for it today."
    text2 = "The future belongs to those who prepare for it today; education is their passport."

    # Compute readability metrics
    metrics_text1 = get_readability_metrics(text1)
    metrics_text2 = get_readability_metrics(text2)

    # Compare metrics
    differences = compare_metrics(metrics_text1, metrics_text2)
    similarity = cosine_similarity(metrics_text1, metrics_text2)

    # Print results
    print("\n=== Readability Metrics for Text ===")
    print(f"{'Metric':30} | {'Text 1':>8} | {'Text 2':>8} | {'Diff':>8}")
    for key in sorted(metrics_text1):
        val1 = metrics_text1[key]
        val2 = metrics_text2[key]
        diff = abs(val1 - val2)  # signed difference
        print(f"{key.upper():30} | {val1:8.2f} | {val2:8.2f} | {diff:8.2f}")

    print("\n=== Absolute Differences ===")

    print(f"\nCosine Similarity between texts: {similarity:.4f}")
