# Student Social Profile Segmentation Using Clustering

An unsupervised learning project that explores student social-network profiles and groups students based on their recorded interests.

## Project Overview

The dataset contains **15,000 student profiles** and **40 columns**, consisting of:

- **4 profile attributes:** `gradyear`, `gender`, `age`, and `NumberOffriends`.
- **36 interest features:** non-negative integer counts representing keyword occurrences across different topics.

The 36 interest features are the primary inputs for clustering. Profile attributes are reserved for interpreting and describing the resulting clusters rather than serving as the main clustering inputs.

The data-preparation stage will compare two feature representations:

1. **Count-based representation:** apply a `log1p` transformation, followed by standardization.
2. **Binary representation:** convert interest counts into presence/absence values (`0` or `1`).

These are planned experiments. The final feature representation and clustering algorithm will be selected after model evaluation.

## Project Objectives

1. Explore the dataset structure, data quality, feature distributions, sparsity, and relationships.
2. Prepare and validate suitable feature representations for clustering.
3. Experiment with clustering algorithms and evaluate the resulting segments.
4. Interpret cluster characteristics using the available student profile attributes.

## Dataset

**Source:** [Students Social Network Profile Clustering — Kaggle](https://www.kaggle.com/datasets/zabihullah18/students-social-network-profile-clustering)

**Expected file name:** `03_Clustering_Marketing.csv`

Download the dataset from Kaggle and place the CSV file inside the `data/` directory:

```text
data/03_Clustering_Marketing.csv
```

The dataset is not included in this repository. Check the dataset's license and sharing terms before uploading or redistributing the CSV file on GitHub.

## Repository Structure

```text
student-social-profile-segmentation/
├── data/
│   └── 03_Clustering_Marketing.csv   # Source dataset (download separately)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_data_preparation.ipynb     # Data cleaning and feature preparation
│   └── 03_model_development.ipynb    # Clustering experiments and evaluation
│
├── outputs/
│   ├── figures/                      # Exported plots (PNG, etc.)
│   ├── tables/                       # EDA and experiment summaries (CSV)
│   └── preprocessing/                # Prepared data outputs, when needed
│
├── models/                           # Saved model artifacts (e.g. .joblib)
├── reports/                          # Project reports (PDF, DOCX, etc.)
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `outputs/`, `models/`, and `reports/` directories may be empty at the beginning of the project. Git does not track empty directories; add a `.gitkeep` file when you need an empty directory to appear in the repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR-USERNAME>/student-social-profile-segmentation.git
cd student-social-profile-segmentation
```

Replace `<YOUR-USERNAME>` with the GitHub username or organization that owns the repository.

### 2. Create a virtual environment

Use a Python version supported by the packages listed in `requirements.txt`. The current local development environment uses Python 3.14.7. After installing dependencies in a clean virtual environment, verify that the required libraries import successfully before running the notebooks.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, you can allow it for the current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place `03_Clustering_Marketing.csv` inside the `data/` directory. When running a notebook with the repository root as its working directory, load the dataset with:

```python
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/03_Clustering_Marketing.csv")
df = pd.read_csv(DATA_PATH)
```

When running on Kaggle Notebooks, use the dataset path mounted by Kaggle instead.

### 5. Select the project environment in Jupyter or VS Code

Make sure the notebook kernel or Python interpreter points to the project's `.venv` environment. If the environment is not listed as a notebook kernel, register it with:

```bash
python -m ipykernel install --user --name student-social-profile-segmentation --display-name "Python (.venv - Student Segmentation)"
```

### 6. Run the notebooks in order

1. `01_eda.ipynb` — explore the dataset and document EDA findings.
2. `02_data_preparation.ipynb` — clean the data, prepare feature representations, and validate the prepared data.
3. `03_model_development.ipynb` — run clustering experiments, evaluate models, and interpret clusters.

## Outputs and Model Artifacts

- Exported EDA figures should be saved in `outputs/figures/`.
- EDA summaries and experiment tables should be saved in `outputs/tables/`.
- Prepared datasets or preprocessing outputs should be saved in `outputs/preprocessing/` when needed.
- Final model files and related artifacts should be saved in `models/` (for example, `.joblib` files).
- Project documentation should be saved in `reports/`.

Only commit generated outputs that are useful for reviewing, reproducing, or presenting the project. Check file sizes before committing large datasets or model artifacts.

## Model Evaluation

The model-development stage may use the following internal clustering metrics:

- **Silhouette Score:** evaluates how well observations fit within their assigned clusters compared with other clusters. Higher values are generally preferred.
- **Davies–Bouldin Index:** evaluates similarity between clusters. Lower values are generally preferred.
- **Calinski–Harabasz Index:** compares between-cluster dispersion with within-cluster dispersion. Higher values are generally preferred.

These metrics should be considered together with cluster sizes, feature profiles, and the practical interpretability of the segments. No single metric should determine the final model by itself.

## Project Notes

- Keep the original dataset unchanged.
- Record data-cleaning and feature-transformation decisions in the notebooks.
- The dataset does not provide a unique student identifier. Identical interest patterns do not necessarily represent duplicate student identities, and full-row duplicates should be reviewed before removal.
- Profiles with all-zero interest features mean that no interests were recorded in the available 36 features; they do not prove that a student has no interests.
- Save final model artifacts in `models/` and project documents in `reports/` when needed.
- Only upload the dataset and model files to GitHub when their size and applicable license or sharing terms permit it.

## Contributors

| Name |
|---|
| Michael Andreas Tjendra |
| Diky Diwa Suwanto |
| Muhammad Rizki |
| Bagas Arma |
| Andika Dermawan |

## License

Add a license for this project's code before publishing if you intend to make it reusable by others. The code license and the dataset's license are separate; check the Kaggle dataset terms before sharing the dataset file.
