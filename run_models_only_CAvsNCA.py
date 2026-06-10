import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


PROCESSED_DIR = "processed_data"
OUT_DIR = "model_results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_X_y(X_path, y_path, label_column):
    """
    Load prepared X and y files.

    X file:
        rows = samples
        columns = ENSG features

    y file:
        rows = samples
        one column = label
    """
    X = pd.read_csv(X_path, index_col=0)
    y_df = pd.read_csv(y_path, index_col=0)

    if label_column not in y_df.columns:
        raise ValueError(f"Cannot find label column: {label_column}")

    # Make sure X and y have the same sample order.
    missing_in_y = [s for s in X.index if s not in y_df.index]
    if len(missing_in_y) > 0:
        raise ValueError(f"These samples are in X but missing in y: {missing_in_y[:10]}")

    y_df = y_df.loc[X.index]
    y = y_df[label_column].astype(int).values

    print("\nLoaded data:")
    print("X path:", X_path)
    print("y path:", y_path)
    print("X shape:", X.shape)
    print("y length:", len(y))
    print("Label counts:")
    print(pd.Series(y).value_counts())

    return X, y


def make_models(k_features):
    """
    Build baseline classifiers.

    Feature selection is inside the pipeline, so it happens only within
    each training fold during cross-validation.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("remove_zero_variance", VarianceThreshold()),
            ("select_top_features", SelectKBest(score_func=f_classif, k=k_features)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=1
            ))
        ]),

        "LinearSVM": Pipeline([
            ("remove_zero_variance", VarianceThreshold()),
            ("select_top_features", SelectKBest(score_func=f_classif, k=k_features)),
            ("scale", StandardScaler()),
            ("classifier", SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=1
            ))
        ]),

        "GaussianNB": Pipeline([
            ("remove_zero_variance", VarianceThreshold()),
            ("select_top_features", SelectKBest(score_func=f_classif, k=k_features)),
            ("scale", StandardScaler()),
            ("classifier", GaussianNB())
        ]),

        "KNN": Pipeline([
            ("remove_zero_variance", VarianceThreshold()),
            ("select_top_features", SelectKBest(score_func=f_classif, k=k_features)),
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),

        "RandomForest": Pipeline([
            ("remove_zero_variance", VarianceThreshold()),
            ("select_top_features", SelectKBest(score_func=f_classif, k=k_features)),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=1
            ))
        ]),

        "LogisticRegression_L1": Pipeline([
            ("remove_zero_variance", VarianceThreshold()),
            ("select_top_features", SelectKBest(score_func=f_classif, k=k_features)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l1",
                class_weight="balanced",
                solver="liblinear",
                max_iter=10000,
                random_state=1
            ))
        ])
    }

    return models


def evaluate_models(X, y, task_name, k_features=500):
    """
    Run 5-fold stratified cross-validation.
    Save metrics, predictions, confusion matrix, and ROC plot.
    """
    print("\n========================================")
    print("Task:", task_name)
    print("Original X shape:", X.shape)

    # Make sure all values are numeric.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Use log2(TPM + 1). If prepare_data.py already did log transform,
    # remove this line to avoid double log transform.
    # X = np.log2(X + 1)

    k_features = min(k_features, X.shape[1])
    print("k_features:", k_features)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    models = make_models(k_features)

    rows = []
    plt.figure(figsize=(6, 5))

    for model_name, model in models.items():
        print(f"\nRunning model: {model_name}")

        prob = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        pred = (prob >= 0.5).astype(int)

        auc = roc_auc_score(y, prob)
        acc = accuracy_score(y, pred)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

        rows.append({
            "task": task_name,
            "model": model_name,
            "k_features": k_features,
            "AUC": auc,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

        pred_df = pd.DataFrame({
            "sample": X.index,
            "true_label": y,
            "predicted_probability": prob,
            "predicted_label": pred
        })

        pred_df.to_csv(
            os.path.join(OUT_DIR, f"{task_name}_{model_name}_predictions.csv"),
            index=False
        )

        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, label=f"{model_name}, AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(task_name)
    plt.legend(fontsize=8)
    plt.tight_layout()

    roc_path = os.path.join(OUT_DIR, f"{task_name}_ROC.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    result_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)

    metrics_path = os.path.join(OUT_DIR, f"{task_name}_metrics.csv")
    result_df.to_csv(metrics_path, index=False)

    print("\nResult:")
    print(result_df)
    print("Saved metrics:", metrics_path)
    print("Saved ROC:", roc_path)

    return result_df


def main():
    # Task 1: cancer vs normal
    X1, y1 = load_X_y(
        X_path=os.path.join(PROCESSED_DIR, "X_cancer_vs_normal.csv"),
        y_path=os.path.join(PROCESSED_DIR, "y_cancer_vs_normal.csv"),
        label_column="cancer_label"
    )

    result1 = evaluate_models(
        X=X1,
        y=y1,
        task_name="cancer_vs_normal",
        k_features=500
    )


    # Task 2: recurrence vs non-recurrence
    X2, y2 = load_X_y(
        X_path=os.path.join(PROCESSED_DIR, "X_recurrence.csv"),
        y_path=os.path.join(PROCESSED_DIR, "y_recurrence.csv"),
        label_column="recurrence_label"
    )

    result2 = evaluate_models(
        X=X2,
        y=y2,
        task_name="recurrence_vs_nonrecurrence",
        k_features=500
    )

    all_results = pd.concat([result1, result2], axis=0)
    all_results_path = os.path.join(OUT_DIR, "all_metrics_summary.csv")
    all_results.to_csv(all_results_path, index=False)

    print("\nDONE.")
    print("Final summary:", all_results_path)


if __name__ == "__main__":
    main()