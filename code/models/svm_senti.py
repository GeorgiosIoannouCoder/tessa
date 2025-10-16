import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline as sk_make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, f1_score
)

LABEL_NAMES = ["negative","neutral","positive"]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _build_pipeline(use_smote: bool, seed: int, min_df: int, ngram_max: int, svd_components: int):
    tfidf = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, strip_accents="unicode")
    base = LinearSVC(random_state=seed, class_weight="balanced")
    clf = CalibratedClassifierCV(base)

    if not use_smote:
        return sk_make_pipeline(tfidf, clf)

    # SMOTE path: need dense vectors via SVD; import lazily to avoid hard dependency
    from sklearn.decomposition import TruncatedSVD
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import make_pipeline as im_make_pipeline
    return im_make_pipeline(tfidf, TruncatedSVD(n_components=svd_components, random_state=seed),
                            SMOTE(random_state=seed), clf)

def train_eval(*, limit=None, save_dir="models_performance/svm", seed=42,
               use_smote=False, svd_components=300, min_df=2, ngram_max=2):
    _ensure_dir(save_dir)

    print("Step 1: loading dataset…", flush=True)
    ds = load_dataset("tweet_eval", "sentiment")
    X, y = ds["train"]["text"], ds["train"]["label"]
    if limit is not None:
        X, y = X[:limit], y[:limit]

    print("Step 2: splitting…", flush=True)
    test_size = 4000 if limit is None else max(100, int(0.25 * len(X)))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    print("Step 3: vectorizing + fitting (with calibration)…", flush=True)
    pipe = _build_pipeline(use_smote=use_smote, seed=seed, min_df=min_df, ngram_max=ngram_max,
                           svd_components=svd_components)
    pipe.fit(X_tr, y_tr)

    print("Step 4: scoring + plotting…", flush=True)
    y_pred = pipe.predict(X_te)
    report = classification_report(y_te, y_pred, target_names=LABEL_NAMES)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    print(report)
    print(f"\nMacro-F1: {macro_f1:.3f}")

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report + f"\nMacro-F1: {macro_f1:.3f}\n")
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({"macro_f1": float(macro_f1)}, f)

    cm = confusion_matrix(y_te, y_pred, labels=[0,1,2])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(["neg","neu","pos"]); ax.set_yticklabels(["neg","neu","pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("SVM Sentiment — TF-IDF + LinearSVC (calibrated)")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200); plt.close(fig)

    proba = pipe.predict_proba(X_te)
    y_true = np.array(y_te)
    pr_paths = []
    for k, name in enumerate(["neg","neu","pos"]):
        fig2, ax2 = plt.subplots()
        y_bin = (y_true == k).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, proba[:, k])
        ap = average_precision_score(y_bin, proba[:, k])
        ax2.plot(rec, prec)
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title(f"{name} PR curve (AP={ap:.3f})")
        fig2.tight_layout()
        p = os.path.join(save_dir, f"pr_curve_class_{k}.png")
        plt.savefig(p, dpi=200); plt.close(fig2)
        pr_paths.append(p)

    return {"macro_f1": macro_f1, "cm_path": cm_path, "pr_paths": pr_paths}

def _parse_args():
    ap = argparse.ArgumentParser(description="SVM sentiment (TweetEval)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default="models_performance/svm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-smote", action="store_true")
    ap.add_argument("--svd-components", type=int, default=300)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=2)
    return ap.parse_args()

def main():
    a = _parse_args()
    train_eval(limit=a.limit, save_dir=a.save_dir, seed=a.seed,
               use_smote=a.use_smote, svd_components=a.svd_components,
               min_df=a.min_df, ngram_max=a.ngram_max)

if __name__ == "__main__":
    main()
