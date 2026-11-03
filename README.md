Student Social Profile Segmentation Using Clustering
An unsupervised learning project that explores student social-network profiles and groups students based on their recorded interests.
Project Overview
The dataset contains 15,000 student profiles and 40 columns:
- 4 profile attributes: gradyear, gender, age, and NumberOffriends.
- 36 interest features: non-negative integer counts representing keyword occurrences across different topics.
The project uses the 36 interest features as the primary input for clustering. Profile attributes are intended for interpreting and describing the resulting clusters, rather than serving as the primary clustering input.

The data-preparation stage will compare two feature representations:
1. Count-based: log1p transformation followed by standardization.
2. Binary-based: presence/absence representation (0 or 1).
These are planned experiments. The final representation and clustering algorithm will be selected after model evaluation.
Project Objectives
1. Explore the dataset structure, data quality, feature distributions, sparsity, and relationships.
2. Prepare and validate suitable feature representations for clustering.
3. Experiment with clustering algorithms and evaluate the resulting segments.
4. Interpret cluster characteristics using the available student profile attributes.
Dataset
Source: Students Social Network Profile Clustering — Kaggle

Expected CSV filename:
03_Clustering_Marketing.csv
Download the dataset from Kaggle and place it directly inside the data/ folder:
data/03_Clustering_Marketing.csv
The dataset is not bundled with this repository. Check the dataset's license and terms before uploading or redistributing the CSV on GitHub.
Repository Structure
student-social-profile-segmentation/
├── data/
│   └── 03_Clustering_Marketing.csv   # Dataset
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_data_preparation.ipynb     # Data cleaning and transformation
│   └── 03_model_development.ipynb    # Clustering experiments and evaluation
├── models/                           # Saved model artifacts (.joblib)
├── reports/                          # Project reports (.docx, .pdf, etc.)
├── .gitignore
├── requirements.txt
└── README.md
Getting Started
1. Clone the repository
git clone https://github.com/<YOUR-USERNAME>/student-social-profile-segmentation.git
cd student-social-profile-segmentation
Replace <YOUR-USERNAME> with the GitHub username or organization that owns the repository.
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Add the dataset
Place 03_Clustering_Marketing.csv inside the data/ folder. In a notebook running from the repository root, use:
from pathlib import Path

DATA_PATH = Path("data/03_Clustering_Marketing.csv")
df = pd.read_csv(DATA_PATH)
When running on Kaggle Notebooks, use the dataset path mounted by Kaggle instead.
5. Run the notebooks in order
1. 01_eda.ipynb — explore the dataset and document EDA findings.
2. 02_data_preparation.ipynb — clean the data, prepare feature representations, and validate the prepared data.
3. 03_model_development.ipynb — run clustering experiments, evaluate models, and interpret clusters.
Model Evaluation
The model-development stage may use the following internal clustering metrics:
- Silhouette Score: evaluates how well observations fit within their assigned clusters compared with other clusters.
- Davies–Bouldin Index: evaluates similarity between clusters; lower values are generally preferred.
- Calinski–Harabasz Index: compares between-cluster dispersion with within-cluster dispersion; higher values are generally preferred.
The metrics should be considered together with cluster sizes, feature profiles, and the practical interpretability of the segments. No single metric should determine the final model by itself.
Project Notes
- Keep the original dataset unchanged.
- Record data-cleaning and feature-transformation decisions in the notebooks.
- Preserve the distinction between identical interest patterns and confirmed duplicate student identities. The dataset does not provide a unique student identifier.
- Save final model artifacts in models/ and project documents in reports/ when needed.
- Only upload the dataset and model files to GitHub when their size and applicable license or sharing terms permit it.
Contributors
Name
Michael Andreas Tjendra
Diky Diwa Suwanto
Enjel Boyzest Saragih


License
Add a license for this project's code before publishing if you intend to make it reusable by others. The code license and the dataset's license are separate; check the Kaggle dataset terms before sharing the dataset file.