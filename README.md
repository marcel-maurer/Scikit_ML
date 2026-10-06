# Scikit_ML

Python portfolio covering **machine learning workflows** and **local AI/LLM engineering**.

This repository documents my hands-on development from classical scikit-learn projects to a local GGUF/llama.cpp agent project. The focus is practical implementation: preprocessing, evaluation and reusable ML pipelines on one side, and local model initialization, GPU offloading, multimodal serving and agent tooling on the other.

## Featured Projects

| Project | Focus | Implementation |
| --- | --- | --- |
| **Brunno Local AI Agent** | Local LLMs, GGUF, llama.cpp, GPU offloading, vision models, agent tools | [AI_Agent](AI_Agent/) |
| **Titanic Survival Prediction** | Binary classification, preprocessing, model comparison | [Titanic_Disaster.py](Projects/RefactoredVersions/Titanic_Disaster.py) |
| **Credit Fraud Detection** | Imbalanced classification, evaluation, SMOTE-capable pipelines | [CreditFraud.py](Projects/RefactoredVersions/CreditFraud.py) |
| **Bank Churn Prediction** | Binary classification, preprocessing, model benchmarking | [BankChurn.py](Projects/RefactoredVersions/BankChurn.py) |
| **Original ML implementation** | Earlier project work and experimentation | [OwnVersion.ipynb](Projects/OwnVersion.ipynb) |

## Brunno — Local AI / LLM Engineering

> **Work in progress:** Brunno is an ongoing personal development project and is not finished yet. The architecture, features and code are actively being developed, tested and refactored as I continue learning and expanding the system.

Brunno is my experimental local AI-agent project. It expands this portfolio beyond classical ML and explores running and integrating modern language and vision models locally.

The current code includes:

- local GGUF inference with `llama-cpp-python`
- llama.cpp server management from Python
- GPU-layer configuration and fallback behavior for CUDA/VRAM failures
- configurable context size
- Qwen text-model integration
- Qwen-VL / multimodal model serving with an mmproj adapter
- OpenAI-compatible local chat requests
- external personality and memory context
- simple memory extraction and JSON tooling
- subprocess-based local infrastructure

The public portfolio version intentionally excludes model weights, credentials and private user/memory files. Local machine paths are configurable through environment variables instead of being tied to my development workstation.

**Project documentation:** [AI_Agent/README.md](AI_Agent/README.md)

## Machine Learning Focus

The refactored ML scripts use configurable workflows built around scikit-learn and imbalanced-learn. Depending on the project/configuration, the code includes:

- train/test splitting
- numerical and categorical preprocessing
- scikit-learn / imbalanced-learn pipelines
- cross-validation
- randomized hyperparameter search
- model comparison
- optional SMOTE
- result/model persistence
- optional XGBoost support

Models represented in the codebase include Logistic Regression, Linear SVM, Decision Tree, Random Forest and XGBoost.

## Technical Stack

**AI / LLM**
- llama.cpp
- llama-cpp-python
- GGUF models
- Qwen / Qwen-VL
- local OpenAI-compatible endpoints
- multimodal model adapters
- GPU offloading / CUDA-oriented configuration

**Machine Learning & Data**
- scikit-learn
- imbalanced-learn
- pandas
- NumPy
- SciPy
- XGBoost
- preprocessing and feature-engineering pipelines
- cross-validation and model evaluation

**Development**
- Python
- object-oriented design
- dataclasses and configuration objects
- subprocess/process management
- JSON-based configuration/context
- Jupyter Notebook
- Git and GitHub
- Linux development environment

## Repository Structure

```text
Scikit_ML/
├── AI_Agent/
│   ├── AgentInits/
│   │   ├── __init__.py
│   │   ├── AgentConfigs.py
│   │   ├── Agent_Presets.py
│   │   ├── Agent_tools.py
│   │   ├── external_subprocess_run.py
│   │   └── init_model.py
│   └── README.md
├── CSVData/
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

## Getting Started

```bash
git clone https://github.com/marcel-maurer/Scikit_ML.git
cd Scikit_ML

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the Brunno project, model files are not part of the repository. Configure your local locations before running it:

```bash
export BRUNNO_HOME="$HOME/Brunno"
export LLAMA_CPP_HOME="$HOME/llama.cpp"
```

For the classical ML scripts, check the individual `Config` section and adapt the dataset path/target to the local dataset.

## Development Approach

I use this repository to document both working results and the development process behind them. Earlier implementations remain visible where useful, while later versions show refactoring, debugging and attempts to turn repeated logic into reusable components.

My current direction is moving from classical ML workflows toward complete AI applications: local inference, multimodal models, tool integration, memory/context systems, APIs and reusable agent architecture.

## AI-Assisted Development

AI tools are part of my learning and development workflow for explanations, debugging, reviewing approaches and refactoring. AI-generated suggestions are not treated as finished solutions: I test, debug, adapt and work through the resulting implementation.

---

**Author:** Marcel Maurer  
**Focus:** Python · Machine Learning · AI Engineering · Local LLMs
