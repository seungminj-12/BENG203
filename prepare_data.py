import os
import numpy as np
import pandas as pd

RAW_DIR = "data"
OUT_DIR = "processed_data"
os.makedirs(OUT_DIR, exist_ok=True)


def load_pnas_bc_tpm(path):
    """
    Load PNAS breast cancer TPM matrix.

    Original file:
    - no header
    - first column = ENSG feature ID
    - 96 sample columns without sample names

    Output:
    - samples x features
    - sample names = S01-S96
    """
    df = pd.read_csv(path, sep="\t", header=None, index_col=0)

    sample_names = [f"S{i:02d}" for i in range(1, df.shape[1] + 1)]
    df.columns = sample_names

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.T
    X.index.name = "sample"
    X.columns = X.columns.astype(str)

    return X


def load_pnas_normal_tpm(path):
    """
    Load PNAS normal TPM matrix.

    Original file:
    - first row = normal sample names N1-N32
    - first column = ENSG feature ID, but this column has no header name

    Output:
    - samples x features
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.T
    X.index.name = "sample"
    X.columns = X.columns.astype(str)

    return X


def load_patient_info(path):
    """
    Load PNAS patient metadata.
    Extract sample_short from sample_id.
    Example:
    S01_B14 -> S01
    """
    meta = pd.read_csv(path)

    if "sample_id" not in meta.columns:
        raise ValueError("sample_id column is missing from pnas_patient_info.csv")

    if "recurStatus" not in meta.columns:
        raise ValueError("recurStatus column is missing from pnas_patient_info.csv")

    meta = meta.copy()

    meta["sample_short"] = meta["sample_id"].astype(str).str.extract(r"(S\d+)")[0]

    meta["recurrence_label"] = (
        meta["recurStatus"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"R": 1, "N": 0})
    )

    if meta["sample_short"].isna().any():
        bad = meta.loc[meta["sample_short"].isna(), ["sample_id"]]
        print(bad)
        raise ValueError("Some sample_id values do not contain Sxx pattern.")

    if meta["recurrence_label"].isna().any():
        bad = meta.loc[meta["recurrence_label"].isna(), ["sample_id", "recurStatus"]]
        print(bad)
        raise ValueError("Some recurStatus values are not R or N.")

    return meta


def save_cancer_vs_normal_data(X_bc, X_normal):
    """
    Prepare cancer vs normal classification data.

    Label:
    cancer = 1
    normal = 0
    """
    shared_features = X_bc.columns.intersection(X_normal.columns)

    X_bc_shared = X_bc[shared_features].copy()
    X_normal_shared = X_normal[shared_features].copy()

    X = pd.concat([X_bc_shared, X_normal_shared], axis=0)

    y = pd.DataFrame({
        "sample": X.index,
        "group": ["cancer"] * X_bc_shared.shape[0] + ["normal"] * X_normal_shared.shape[0],
        "label": [1] * X_bc_shared.shape[0] + [0] * X_normal_shared.shape[0]
    })

    # Add label into a viewable matrix for checking only
    labeled_matrix = X.copy()
    labeled_matrix.insert(0, "group", y["group"].values)
    labeled_matrix.insert(1, "label", y["label"].values)

    X.to_csv(os.path.join(OUT_DIR, "X_cancer_vs_normal.csv"))
    y.to_csv(os.path.join(OUT_DIR, "y_cancer_vs_normal.csv"), index=False)
    labeled_matrix.to_csv(os.path.join(OUT_DIR, "labeled_cancer_vs_normal_matrix.csv"))

    print("\nCancer vs normal data saved.")
    print("X_cancer_vs_normal shape:", X.shape)
    print("y_cancer_vs_normal label counts:")
    print(y["label"].value_counts())
    print("Number of shared features:", len(shared_features))

    print("\nCancer vs normal label check:")
    print(y.head())
    print(y.tail())


def save_recurrence_data(X_bc, meta):
    """
    Prepare recurrence vs non-recurrence data.

    Label:
    recurrence R = 1
    non-recurrence N = 0

    Match expression samples S01-S96 to metadata sample_id like S01_B14.
    """
    meta_small = meta[["sample_short", "sample_id", "recurStatus", "recurrence_label"]].copy()
    meta_small = meta_small.set_index("sample_short")

    missing = [s for s in X_bc.index if s not in meta_small.index]
    if len(missing) > 0:
        raise ValueError(f"These expression samples are missing from metadata: {missing}")

    meta_ordered = meta_small.loc[X_bc.index]

    y = pd.DataFrame({
        "sample": X_bc.index,
        "sample_id": meta_ordered["sample_id"].values,
        "recurStatus": meta_ordered["recurStatus"].values,
        "label": meta_ordered["recurrence_label"].values
    })

    labeled_matrix = X_bc.copy()
    labeled_matrix.insert(0, "sample_id", y["sample_id"].values)
    labeled_matrix.insert(1, "recurStatus", y["recurStatus"].values)
    labeled_matrix.insert(2, "label", y["label"].values)

    X_bc.to_csv(os.path.join(OUT_DIR, "X_recurrence.csv"))
    y.to_csv(os.path.join(OUT_DIR, "y_recurrence.csv"), index=False)
    labeled_matrix.to_csv(os.path.join(OUT_DIR, "labeled_recurrence_matrix.csv"))

    print("\nRecurrence data saved.")
    print("X_recurrence shape:", X_bc.shape)
    print("y_recurrence label counts:")
    print(y["label"].value_counts())

    print("\nRecurrence label check:")
    print(y.head())
    print(y.tail())


def main():
    X_bc = load_pnas_bc_tpm(os.path.join(RAW_DIR, "pnas_tpm_96_nodup.txt"))
    X_normal = load_pnas_normal_tpm(os.path.join(RAW_DIR, "pnas_normal_tpm.txt"))
    meta = load_patient_info(os.path.join(RAW_DIR, "pnas_patient_info.csv"))

    print("Loaded breast cancer TPM:", X_bc.shape)
    print("Loaded normal TPM:", X_normal.shape)
    print("Loaded patient metadata:", meta.shape)

    print("\nBreast cancer sample names:")
    print(X_bc.index[:5].tolist(), "...", X_bc.index[-5:].tolist())

    print("\nNormal sample names:")
    print(X_normal.index[:5].tolist(), "...", X_normal.index[-5:].tolist())

    print("\nFirst 5 feature IDs in breast cancer:")
    print(X_bc.columns[:5].tolist())

    print("\nFirst 5 feature IDs in normal:")
    print(X_normal.columns[:5].tolist())

    # log2(TPM + 1)
    X_bc_log = np.log2(X_bc + 1)
    X_normal_log = np.log2(X_normal + 1)

    save_cancer_vs_normal_data(X_bc_log, X_normal_log)
    save_recurrence_data(X_bc_log, meta)

    print("\nDONE. Processed files are saved in:", OUT_DIR)
    print("Main files for model script:")
    print("1. processed_data/X_cancer_vs_normal.csv")
    print("2. processed_data/y_cancer_vs_normal.csv")
    print("3. processed_data/X_recurrence.csv")
    print("4. processed_data/y_recurrence.csv")


if __name__ == "__main__":
    main()