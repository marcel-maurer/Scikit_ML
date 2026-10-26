# Scikit_ML

**Python projects in machine learning, data analysis and local LLM integration.**

I am Marcel Maurer, a self-taught Python and machine learning developer. In this repository, I work on classification problems, compare models and build configurable evaluation workflows. Alongside the ML projects, I am developing **BRUNNO**, a personal project for running and integrating language and vision models locally.

## Start with these projects

| Project | What to look at | Code |
| --- | --- | --- |
| Titanic survival prediction | Mixed numerical/categorical data, preprocessing pipelines and binary classification | [Titanic_Disaster.py](Projects/RefactoredVersions/Titanic_Disaster.py) |
| Credit card fraud detection | Imbalanced classification, optional oversampling and precision/recall evaluation | [CreditFraud.py](Projects/RefactoredVersions/CreditFraud.py) |
| Bank customer churn | Configurable model comparisons and CSV/ZIP dataset loading | [BankChurn.py](Projects/RefactoredVersions/BankChurn.py) |
| ML experimentation notebook | Data exploration, classification experiments and evaluation plots | [OwnVersion.ipynb](Projects/OwnVersion.ipynb) |
| BRUNNO | Local GGUF models, llama.cpp processes, model configuration and context handling | [AI_Agent](AI_Agent/) |

## Machine learning workflow

The three classification scripts share a configurable workflow:

1. Load a CSV or a selected CSV from a ZIP archive and validate the binary target.
2. Create a stratified training/test split.
3. Fit numerical and categorical preprocessing inside the model pipeline.
4. Compare classifiers using cross-validation and optional randomized hyperparameter search.
5. Evaluate the fitted models on the held-out test split.
6. Export benchmark results and optionally save fitted pipelines.

### Preprocessing and sampling

- Numerical features: mean imputation and MinMax scaling.
- Categorical features: most-frequent imputation and one-hot encoding that handles unseen categories.
- Optional SMOTE, SMOTENC or SMOTEN, selected according to the feature types.
- Sampling and learned preprocessing run inside the training pipeline, including during cross-validation.
- A custom `DataTransform` step provides an extension point for feature engineering. In the current classification scripts it returns an unchanged copy, so the `base` and `transformed` branches do not yet represent different feature sets.

### Models and evaluation

The scripts compare Logistic Regression, Linear SVM, Decision Tree and Random Forest, with XGBoost included when installed.

Evaluation records **F1, precision, recall and accuracy**, along with training scores and cross-validation standard deviations. When cross-validation and hyperparameter search are both enabled, the search runs inside each outer training fold. Inner search scores, outer validation scores and holdout test scores are recorded separately.

The holdout scores should be used for final reporting after model choices have been made from training-data validation. Repeatedly choosing models or features based on the holdout results would compromise that separation.

Results are written to `BenchmarkResults/<model_name>_results.csv` when enabled. Optional model saving uses `joblib` and stores fitted pipelines under `ModelSaveData/<model_name>/`.

## Run an ML project

From a terminal on Linux:

```bash
git clone https://github.com/marcel-maurer/Scikit_ML.git
cd Scikit_ML
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn imbalanced-learn joblib matplotlib seaborn tqdm
python Projects/RefactoredVersions/Titanic_Disaster.py
```

This installs the dependencies for the classical ML scripts. XGBoost is optional:

```bash
python -m pip install xgboost
```

The full [requirements.txt](requirements.txt) also includes the local AI dependencies, such as PyTorch and `llama-cpp-python`. BRUNNO needs its own local model files and llama.cpp setup.

### Dataset configuration

| Script | Default input | Target | Availability |
| --- | --- | --- | --- |
| `Titanic_Disaster.py` | `CSVData/TitanicDisaster/titanic.csv` | `Survived` | Included |
| `BankChurn.py` | `CSVData/bankChurn/bankChurncsv.zip` | `Exited` | Included |
| `CreditFraud.py` | `CSVData/CreditFraud/archive.zip` | `Class` | Download separately; see [dataset link](CSVData/CreditFraud/credit_fraud_link.txt) |

Run the other scripts from the repository root:

```bash
python Projects/RefactoredVersions/BankChurn.py
python Projects/RefactoredVersions/CreditFraud.py
```

Each script has a `Config` dataclass and a `main()` function. Adjust dataset paths, target names, fold counts, search iterations and sampling settings there. Relative dataset paths are resolved from the repository root. For ZIP files containing multiple matching CSVs, set `csv_member` explicitly.

The default scripts enable cross-validation and hyperparameter search, which can take time, especially on the fraud dataset. For an initial run, set `use_grid=False` and `compare_feature_engineering=False` in `main()`; reduce `n_jobs` if needed.

To use the notebook, install JupyterLab, open [OwnVersion.ipynb](Projects/OwnVersion.ipynb), set `CONFIG.csv_path` to your dataset and run the cells in order. Its default path still points to my local development machine.

## BRUNNO — local AI project

**Status: in development.**

BRUNNO explores how to connect local inference, model configuration and application logic in Python. The published modules include:

- GGUF inference through `llama-cpp-python`.
- Python subprocess management for text and vision llama.cpp servers.
- Configurable context length and GPU layer offloading, with retry logic for resource failures.
- Qwen text-model and Qwen-VL vision-model configuration, including an mmproj adapter.
- Requests to a local OpenAI-compatible chat endpoint.
- Loading external personality/context files and simple JSON-based memory handling.

The modules are experimental and still need integration work; the repository does not currently provide a complete desktop application or a finished agent framework.

Local paths are configured through `AgentConfigs.py` and environment variables:

```bash
export BRUNNO_HOME="$HOME/Brunno"
export LLAMA_CPP_HOME="$HOME/llama.cpp"
```

Model weights and private context files are not included. See [AI_Agent/README.md](AI_Agent/README.md) for the module structure and configuration notes.

## Repository guide

| Location | Contents |
| --- | --- |
| [Projects/RefactoredVersions](Projects/RefactoredVersions/) | Configurable classification scripts |
| [Projects/OwnVersion.ipynb](Projects/OwnVersion.ipynb) | Interactive ML experiments |
| [AI_Agent](AI_Agent/) | BRUNNO local LLM modules |
| [CSVData](CSVData/) | Included datasets and external dataset references |
| [MLUtilitys/MLUtilitys.py](MLUtilitys/MLUtilitys.py) | Experimental ML code generator |
| [courses/Udemy](courses/Udemy/) | Course certificates and learning records |

## Tools used

**ML and data:** Python, pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib and optional XGBoost.

**Local AI:** llama.cpp, llama-cpp-python, GGUF, PyTorch, Qwen/Qwen-VL, HTTP requests and subprocess management.

**Development:** Linux, Git, GitHub and Jupyter.

## Development approach

I use these projects to practise data preparation, model evaluation and Python application development. My next steps are dataset-specific feature engineering, clearer experiment summaries and continued integration of BRUNNO's local-model components.

AI tools support my learning, debugging, code review and refactoring workflow.

**Marcel Maurer · Python · Machine Learning · Local AI**
