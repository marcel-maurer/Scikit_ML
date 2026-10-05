# Scikit_ML

Machine learning portfolio focused on practical **Python**, **scikit-learn**, data preprocessing, model evaluation, and reusable ML workflows.

This repository documents my hands-on development in machine learning: from exploratory notebook work to more structured Python implementations with preprocessing pipelines, cross-validation, hyperparameter search, class-imbalance handling, and reusable utilities.

> **Portfolio note:** `Projects/OwnVersion.ipynb` represents earlier/original work. The scripts in `Projects/RefactoredVersions/` are later, more structured versions developed through further debugging, refactoring, experimentation, and AI-assisted development.

## Featured Projects

| Project | Focus | Implementation |
| --- | --- | --- |
| **Titanic Survival Prediction** | Binary classification, preprocessing, model comparison | [Titanic_Disaster.py](Projects/RefactoredVersions/Titanic_Disaster.py) |
| **Credit Fraud Detection** | Imbalanced classification, evaluation, SMOTE-capable pipelines | [CreditFraud.py](Projects/RefactoredVersions/CreditFraud.py) |
| **Bank Churn Prediction** | Binary classification, preprocessing, model benchmarking | [BankChurn.py](Projects/RefactoredVersions/BankChurn.py) |
| **Original / learning implementation** | Earlier project work and experimentation | [OwnVersion.ipynb](Projects/OwnVersion.ipynb) |

The refactored scripts use a configurable workflow built around scikit-learn and imbalanced-learn. Depending on the project/configuration, the code includes train/test splitting, numerical and categorical preprocessing, cross-validation, randomized hyperparameter search, model comparison, optional SMOTE, result persistence, and optional XGBoost support.

## Technical Focus

**Machine Learning**
- Classification workflows
- scikit-learn pipelines
- Cross-validation
- Hyperparameter optimization with `RandomizedSearchCV`
- Model comparison
- Imbalanced-data handling with SMOTE
- Accuracy, precision and F1-based evaluation

**Data & preprocessing**
- pandas and NumPy
- Missing-value handling
- Numerical scaling
- Categorical encoding
- Feature-engineering hooks
- Reusable preprocessing components

**Models represented in the current codebase**
- Logistic Regression
- Linear SVM
- Decision Tree
- Random Forest
- XGBoost when installed

**Software development**
- Python
- Object-oriented design
- Dataclasses and configuration objects
- Reusable ML utilities
- Model/result persistence
- Jupyter Notebook
- Git and GitHub

## Repository Structure

```text
Scikit_ML/
├── CSVData/
│   ├── CreditFraud/
│   ├── TitanicDisaster/
│   └── bankChurn/
├── MLUtilitys/
│   └── MLUtilitys.py
├── Projects/
│   ├── OwnVersion.ipynb
│   └── RefactoredVersions/
│       ├── BankChurn.py
│       ├── CreditFraud.py
│       └── Titanic_Disaster.py
├── courses/
│   └── Udemy/
├── .gitignore
├── requirements.txt
└── README.md
```

### `Projects/OwnVersion.ipynb`

Earlier/original implementation and experimentation. I keep this material visible because it shows how my approach developed before later refactoring.

### `Projects/RefactoredVersions/`

More structured implementations created from further work on the original ideas. These versions focus on clearer separation of configuration, data loading, preprocessing, pipelines, validation/search, benchmarking, and persistence.

### `MLUtilitys/`

Reusable and experimental ML tooling. This area reflects my work toward reducing repeated setup code and building more modular workflows.

### `CSVData/`

Datasets or dataset references used by the projects. Some datasets may be provided as links or archives rather than duplicated as large raw files.

### `courses/Udemy/`

Course certificates / learning material. This directory is intentionally separated from my project implementations so that portfolio work and learning resources are distinguishable.

## Getting Started

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/marcel-maurer/Scikit_ML.git
cd Scikit_ML

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The individual project scripts contain project-specific configuration such as dataset paths and target columns. Check the `Config` section of a script before running it and adapt the dataset path to your local environment.

Example:

```bash
python Projects/RefactoredVersions/Titanic_Disaster.py
```

> Some scripts are development/refactoring versions rather than packaged command-line applications. Dataset configuration may therefore need to be adjusted before execution.

## Development Approach

My goal with this repository is not only to train models, but to understand and improve the complete workflow around them:

1. inspect and prepare data,
2. build reproducible preprocessing,
3. compare multiple estimators,
4. evaluate with appropriate metrics and cross-validation,
5. tune models,
6. debug and refactor the implementation,
7. turn repeated logic into reusable components.

This repository is intentionally a learning **and** portfolio repository. It includes both earlier work and later refactored code so that the development process remains visible.

## AI-Assisted Development

I use AI tools as part of my development and learning workflow for debugging, explanations, reviewing approaches, and refactoring.

Where AI assistance has been used, I do not present generated suggestions as independently written code without qualification. I test, debug, adapt, and work through the resulting implementation. The distinction between the earlier/original work and later refactored versions is kept visible in this repository for that reason.

## Current Direction

I am continuing to improve the repository toward cleaner, reusable end-to-end ML projects. Current areas of interest include stronger project packaging, reproducible experiment results, automated evaluation, and connecting classical ML experience with modern AI/LLM application development.

---

**Author:** Marcel Maurer  
**Focus:** Python · Machine Learning · Data Science · AI Development
