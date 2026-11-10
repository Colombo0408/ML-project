# Occupancy Detection

Machine Learning course programming assignment — **Fabio Colombo**, University of Pavia.

Binary classification of room occupancy from ambient sensor readings, as a privacy-friendly alternative to cameras.

## Dataset

8143 records with 5 features (Temperature, Humidity, Light, CO2, HumidityRatio) and a binary `Occupancy` label, already split into:

| Split | Records | Occupied |
|---|---|---|
| Train | 6943 | ~21.5% |
| Validation | 400 | ~20.7% |
| Test | 800 | ~19.4% |

The classes are imbalanced, so models are compared on precision and recall, not only on accuracy. Light and CO2 are by far the strongest predictors. Features are z-score standardized using train-set statistics.

## Models

| Model | Implementation | Hyperparameter search (validation) |
|---|---|---|
| Logistic Regression | PyTorch, SGD + BCE | L1 vs L2 regularization, λ |
| Linear SVM | PyTorch, hinge loss | λ (L2 weight decay) |
| KNN | scikit-learn | k |
| MLP (5 → 16 → 8 → 1, ReLU) | PyTorch, Adam + BCEWithLogits | weight decay |

For each model, ROC and Precision-Recall curves are built on the validation set by sweeping the decision threshold. Three operating points (precision-oriented, recall-oriented, intermediate) are selected and then applied unchanged to the test set.

## Results (test set, intermediate operating point)

| Model | Threshold | Accuracy | Precision | Recall | Val. AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.52 | 0.9912 | 0.9568 | **1.0000** | 0.9961 |
| Linear SVM | 0.62 | 0.9875 | 0.9560 | 0.9806 | **0.9962** |
| KNN (k=5) | 0.40 | 0.9912 | 0.9744 | 0.9806 | 0.9880 |
| MLP | 0.52 | **0.9925** | **0.9806** | 0.9806 | 0.9934 |

The MLP is the best model, but only by a small margin. The linear SVM is consistently the weakest. All models are close, which suggests the data is nearly linearly separable. The full table with all three operating points is in the report.

## Repository structure

```
Data/260831-occupancy.zip                 # train.txt, validation.txt, test.txt
ML_Exam_Occupancy_Detection_v2.ipynb      # final notebook
ML_Exam_Occupancy_Detection_Notebook.pdf  # notebook exported to PDF
Colombo_580655_Occupancy_Detection.pdf    # written report
```

## How to run

1. Copy the folder to `MyDrive/Machine_Learning_Project` on Google Drive.
2. Open `ML_Exam_Occupancy_Detection_v2.ipynb` in Google Colab and run all cells. The notebook mounts Drive and extracts the data zip automatically.

Requirements: `torch`, `scikit-learn`, `matplotlib`, `pandas`.
