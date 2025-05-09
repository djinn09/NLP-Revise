import numpy as np
import textstat
from sklearn.preprocessing import StandardScaler


def get_readability_metrics(text):
    """
    Compute a set of readability metrics for a given text.
    Returns a dict of metric name to value.
    """
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "gunning_fog": textstat.gunning_fog(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        # length-based features
        "difficult_words": textstat.difficult_words(text),
        "sentence_count": textstat.sentence_count(text),
        "avg_sentence_length": textstat.avg_sentence_length(text),
        "syllable_count": textstat.syllable_count(text),
        "lexicon_count": textstat.lexicon_count(text, removepunct=True),
    }


def normalize_metrics(metrics_list):
    """
    Standardize each feature across the corpus to zero mean and unit variance.
    Returns a list of dicts matching input keys.
    """
    keys = list(metrics_list[0].keys())
    mat = np.array([list(m.values()) for m in metrics_list], dtype=float)
    scaler = StandardScaler()
    mat_norm = scaler.fit_transform(mat)
    return [dict(zip(keys, row)) for row in mat_norm]


def euclidean_distance(v1, v2):
    """Compute Euclidean (L2) distance between two feature vectors."""
    arr1 = np.array(list(v1.values()), dtype=float)
    arr2 = np.array(list(v2.values()), dtype=float)
    return float(np.linalg.norm(arr1 - arr2))


def manhattan_distance(v1, v2):
    """Compute Manhattan (L1) distance between two feature vectors."""
    arr1 = np.array(list(v1.values()), dtype=float)
    arr2 = np.array(list(v2.values()), dtype=float)
    return float(np.abs(arr1 - arr2).sum())


def compare_metrics(metrics1, metrics2):
    """
    Compute absolute differences on raw scores for interpretability.
    Returns a dict of metric->abs difference.
    """
    return {k: abs(metrics1[k] - metrics2[k]) for k in metrics1}


if __name__ == "__main__":
    # Example: student vs. model essay
    student_text = "Education is the passport to the future, for tomorrow belongs to those who prepare for it today."
    model_text = "The future belongs to those who prepare for it today; education is their passport."

    # Compute raw metrics
    student_metrics = get_readability_metrics(student_text)
    model_metrics = get_readability_metrics(model_text)

    # Show raw differences for rubric features
    diffs = compare_metrics(student_metrics, model_metrics)

    # Normalize across this pair (expand to larger corpus in real use)
    norm_student, norm_model = normalize_metrics([student_metrics, model_metrics])

    # Compute various distances
    euclid = euclidean_distance(norm_student, norm_model)
    manh = manhattan_distance(norm_student, norm_model)

    # Output results
    print("\n=== Readability Feature Differences ===")
    print(f"{'Metric':30} | {'Student':>8} | {'Model':>8} | {'Diff':>8}")
    print("" + "-" * 60)
    for key in sorted(student_metrics.keys()):
        print(f"{key:30} | {student_metrics[key]:8.2f} | {model_metrics[key]:8.2f} | {diffs[key]:8.2f}")

    print("\n=== Distances on Normalized Readability Features ===")
    print(f"Euclidean distance: {euclid:.4f}")
    print(f"Manhattan distance: {manh:.4f}")

    # Note: In an essay-scoring pipeline, these distances can be combined or fed into a model
