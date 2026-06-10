# BENG203
Machine learning analysis of serum extracellular RNA profiles for breast cancer detection and recurrence prediction.


## Files

### `prepare_data.py`

This script prepares the input data for `run_models_only_CAvsNCA.py`.

After running it, manually rename the label column:

* In `y_cancer_vs_normal.csv`, change the first row of the third column to `cancer_label`
* In `y_recurrence.csv`, change the first row of the third column to `recurrence_label`

### `run_models_only_CAvsNCA.py`

This script runs the cancer-versus-normal classification models.

It uses the full expression matrix as input and selects the top 500 ENSG features using ANOVA F-test within each cross-validation fold.

The script compares multiple classifiers, including Logistic Regression, Linear SVM, Gaussian Naive Bayes, KNN, Random Forest, and L1 Logistic Regression.

## How to Run

```bash
python prepare_data.py
```

Then manually rename the label columns as described above.

```bash
python run_models_only_CAvsNCA.py
```

## Output

Results are saved in the `model_results/` folder, including model metrics, prediction files, and the ROC curve plot.
