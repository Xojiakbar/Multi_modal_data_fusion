# Multi-Modal Physical Exercise Classification

This project notebook studies multi-modal physical exercise classification using the Open MEx dataset. It combines accelerometer readings and depth camera data to classify seven physical exercises in a user-independent setting.

The main notebook is:

- `project-work-MEx-classification.ipynb`

## Overview

The notebook follows a course project workflow for multi-modal data fusion. It prepares paired accelerometer and depth camera samples, extracts features from each modality, trains unimodal and multimodal classifiers, and evaluates the models using confusion matrices and F1 scores.

The dataset contains exercise recordings from multiple subjects. In this project, a subset of 10 subjects is used:

- Subjects 1-7 are used for training.
- Subjects 8-10 are used for testing.
- Each example is created with 5-second windows and 3 seconds of overlap.
- The prepared paired dataset contains 1486 training windows and 598 testing windows.

## Tasks Covered

1. Data preparation, exploration, and visualization
2. Unimodal classification with accelerometer data
3. Unimodal classification with depth camera data
4. Feature-level fusion for multimodal classification
5. Decision-level fusion for multimodal classification
6. Optional person identification task

## Methods

The notebook uses the following techniques:

- Windowing of time-series sensor data
- Standardization and min-max normalization
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Nearest Neighbour classification
- Support Vector Machine with RBF kernel
- Gaussian Naive Bayes
- AdaBoost
- Feature-level fusion
- Decision-level fusion using max, min, product, and sum rules

## Reported Results

Some selected results from the notebook outputs:

| Experiment | Weighted / Overall F1 |
| --- | ---: |
| Accelerometer PCA + LDA + NN | 0.4552 |
| Depth camera PCA + LDA + NN | 0.4337 |
| Feature-level fusion with SVM | 0.6481 |
| Feature-level fusion with Gaussian Naive Bayes | 0.7945 |
| Decision-level fusion: SVM accelerometer + SVM depth, sum rule | 0.6414 |
| Decision-level fusion: AdaBoost accelerometer + SVM depth, product rule | 0.7179 |

## Requirements

The notebook uses Python with common data science libraries:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
```

Install the dependencies with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## Dataset

The notebook expects the MEx dataset files to be available locally. In the submitted notebook, the data location is configured in the first code cell with the variable `loc`.

Update `loc` before running the notebook if your dataset is stored in a different location:

```python
loc = "path/to/MEx"
```

The expected data structure is the MEx folder structure with CSV files organized by subject, exercise, and sensor modality.

## How To Run

1. Open `project-work-MEx-classification.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
2. Make sure the MEx dataset is available.
3. Update the `loc` variable in the first code cell.
4. Run the notebook cells from top to bottom.

## Author

Khojiakbar Botirov
