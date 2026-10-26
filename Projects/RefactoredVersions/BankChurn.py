from __future__ import annotations

import json
import platform
import tqdm
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import joblib
import numpy as np
import pandas as pd
import scipy.stats
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC , LinearSVC
from sklearn.tree import DecisionTreeClassifier


try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

@dataclass
class Config:

    target: str = "Exited"
    csv_path: str = "CSVData/bankChurn/bankChurncsv.zip"

    seed: int = 1234
    test_size: float = 0.20
    use_cv: bool = True
    cross_validation_amount: int = 5

    use_grid: bool = True
    use_tkinter: bool = False
    csv_member: str | None = None
    open_csv_directory: bool = False

    hyperparameter_test_amount: int = 5

    scoring_dict: dict[str, str] = field(default_factory=lambda: {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    })

    scoring_best = "f1"
    scoring_result: str = scoring_best
    refit: str = scoring_best

    use_smote: bool = True
    smote_k_neighbors: int = 5
    compare_feature_engineering: bool = True

    verbose: int = 0
    debug: bool = False
    n_jobs: int = -1

    saving_model: bool = False
    save_results: bool = True
    model_name: str = "BankChurn"
    file_type: str = ".pkl"
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    model_save_path: Path = field(init=False)
    results_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.csv_path = Path(self.csv_path)
        if not self.csv_path.is_absolute():
            self.csv_path = self.base_dir / self.csv_path

        self.model_save_path = (
            self.base_dir / "ModelSaveData" / self.model_name
        )
        self.results_path = (
            self.base_dir
            / "BenchmarkResults"
            / f"{self.model_name}_results.csv"
        )

        self.validate()
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        if self.save_results:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if self.cross_validation_amount < 2:
            raise ValueError("cross_validation_amount must be >= 2.")

        if self.hyperparameter_test_amount < 1:
            raise ValueError("hyperparameter_test_amount must be >= 1.")

        if self.smote_k_neighbors < 1:
            raise ValueError("smote_k_neighbors must be >= 1.")

        if self.scoring_result not in self.scoring_dict:
            raise ValueError(
                f"scoring_result='{self.scoring_result}' "
                f"is not in scoring."
            )

        if self.refit not in self.scoring_dict:
            raise ValueError(
                f"refit='{self.refit}' is not in scoring."
            )

        if not self.file_type.startswith("."):
            raise ValueError("file_type must start with '.'.")

    def print(self) -> None:
        print("=" * 60)
        print("CONFIG")
        print("=" * 60)
        print(f"Dataset:            {self.csv_path}")
        print(f"Target:             {self.target}")
        print(f"Seed:               {self.seed}")
        print(f"Test size:          {self.test_size}")
        print(f"CV folds:           {self.cross_validation_amount}")
        print(f"Search iterations:  {self.hyperparameter_test_amount}")
        print(f"Scoring:            {self.scoring_result}")
        print(f"CV:                 {self.use_cv}")
        print(f"Search:             {self.use_grid}")
        print(f"SMOTE:              {self.use_smote}")
        print(f"Feature comparison: {self.compare_feature_engineering}")
        print(f"Save models:        {self.saving_model}")
        print("=" * 60)

    def open_dir(self, directory: Path | None = None) -> None:
        directory = Path(directory or self.model_save_path)
        if not directory.exists():
            raise FileNotFoundError(directory)

        if platform.system() == "Windows":
            subprocess.Popen(["explorer", str(directory)])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(directory)])
        else:
            subprocess.Popen(["xdg-open", str(directory)])

    def open_csv_dir(self) -> None:

        directory = self.csv_path.parent

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if platform.system() == "Windows":
            subprocess.Popen(["explorer", str(directory)])

        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(directory)])

        else:
            subprocess.Popen(["xdg-open", str(directory)])

    def save_model(self, model: Any, name: str) -> Path | None:
        if not self.saving_model:
            return None

        path = self.model_save_path / f"{name}{self.file_type}"
        joblib.dump(model, path)

        if self.debug:
            print(f"Model saved: {path}")

        return path

@dataclass
class DataContext():
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None

    @property
    def numerical_columns(self) -> list[str]:
        return self.X.select_dtypes(include=np.number).columns.tolist()

    @property
    def categorical_columns(self) -> list[str]:
        return self.X.select_dtypes(exclude=np.number).columns.tolist()

class DataLoader:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.df = None

    def load(self) -> DataContext:
        path = self.config.csv_path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}. Set Config.csv_path to your CSV or ZIP.")
        if path.suffix.lower() == '.zip':
            with ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
                member = self.config.csv_member
                if member is None:
                    matches = []
                    for name in names:
                        with archive.open(name) as stream:
                            if self.config.target in pd.read_csv(stream, nrows=0).columns:
                                matches.append(name)
                    if len(matches) != 1:
                        raise ValueError(f"Set Config.csv_member; matching CSV files: {matches}")
                    member = matches[0]
                with archive.open(member) as stream:
                    df = pd.read_csv(stream)
        else:
            df = pd.read_csv(path)
        if self.config.target not in df:
            raise ValueError(f"Missing target column: {self.config.target}")
        y = df[self.config.target]
        if y.isna().any() or set(y.unique()) != {0, 1}:
            raise ValueError("The target must contain both binary classes 0 and 1, with no missing labels.")
        self.df = df
        return DataContext(df=df, X=df.drop(columns=self.config.target), y=y)

def split_data(data: DataContext, config: Config) -> None:
    (
        data.X_train,
        data.X_test,
        data.y_train,
        data.y_test,
    ) = train_test_split(
        data.X,
        data.y,
        test_size=config.test_size,
        random_state=config.seed,
        shuffle=True,
        stratify=data.y,
    )

class Visualize:
    def __init__(self, data: DataContext) -> None:
        self.data = data.X
        self.target = data.y

    def iqr(self, column: str) -> tuple[float, float, float]:
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        return q1, q3, q3 - q1

    def skew(self, column: str) -> float:
        value = self.data[column].skew()
        print(f"Skewness of '{column}': {value}")
        return value

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        for column in self.data.select_dtypes(
            include=np.number
        ).columns:
            fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=200)
            q1, q3, _ = self.iqr(column)
            mean = self.data[column].mean()

            self.skew(column)

            sns.histplot(self.data, x=column, ax=axes[0])
            axes[0].axvline(q1)
            axes[0].axvline(q3)
            axes[0].axvline(mean)
            axes[0].set_title(f"Histogram: {column}")

            sns.boxplot(self.data, x=column, ax=axes[1])
            axes[1].axvline(q1)
            axes[1].axvline(q3)
            axes[1].axvline(mean)
            axes[1].set_title(f"Boxplot: {column}")

            sns.scatterplot(x=self.data[column], y=self.target, ax=axes[2])
            axes[2].axvline(q1)
            axes[2].axvline(q3)
            axes[2].axvline(mean)
            axes[2].set_title(f"Scatterplot: {column}")

            plt.tight_layout()
            plt.show()

class DataTransform(BaseEstimator, TransformerMixin):
    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "DataTransform":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.new_features(X)

    def new_features(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X.copy()

        return x

def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
                    ("scaler", MinMaxScaler()),
                ]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
    )

def model_variants(config: Config) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            random_state=config.seed,
            max_iter=5000,
            verbose=config.verbose,
        ),
        "LinearSVC": LinearSVC(
            max_iter=10000,
            random_state=config.seed,
            verbose=config.verbose,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_jobs=1,
            random_state=config.seed,
            verbose=config.verbose,
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=config.seed,
        ),
    }

    if XGBClassifier is not None:
        models["XGBClassifier"] = XGBClassifier(
            n_jobs=1,
            random_state=config.seed,
            eval_metric="logloss",
            verbosity=min(config.verbose, 3),
        )

    return models

def model_parameters() -> dict[str, dict[str, Any]]:
    return {
        "LogisticRegression": {
            "estimator__C": scipy.stats.loguniform(1e-4, 1e2),
            "estimator__solver": ["lbfgs"],
        },
        "LinearSVC": {
            "estimator__C": scipy.stats.loguniform(1e-4, 1e2),
            "estimator__loss": ["hinge", "squared_hinge"],
            "estimator__dual": [True],
            "estimator__max_iter": [5000, 10000],
        },
        "SVC": {
            "estimator__C": scipy.stats.loguniform(1e-4, 1e2),
            "estimator__kernel": ["linear", "rbf"],
            "estimator__gamma": ["scale", "auto"],
        },
        "DecisionTreeClassifier": {
            "estimator__criterion": ["gini", "entropy"],
            "estimator__max_depth": [None, 5, 10, 20],
            "estimator__min_samples_split": scipy.stats.randint(2, 20),
            "estimator__min_samples_leaf": scipy.stats.randint(1, 10),
        },
        "RandomForestClassifier": {
            "estimator__n_estimators": scipy.stats.randint(100, 500),
            "estimator__max_depth": scipy.stats.randint(5, 40),
            "estimator__min_samples_split": scipy.stats.randint(2, 20),
            "estimator__min_samples_leaf": scipy.stats.randint(1, 10),
            "estimator__max_features": ["sqrt", "log2"],
            "estimator__bootstrap": [True, False],
        },
        "XGBClassifier": {
            "estimator__n_estimators": scipy.stats.randint(100, 500),
            "estimator__learning_rate": scipy.stats.loguniform(1e-3, 0.3),
            "estimator__max_depth": scipy.stats.randint(3, 10),
            "estimator__subsample": scipy.stats.uniform(0.6, 0.4),
            "estimator__colsample_bytree": scipy.stats.uniform(0.6, 0.4),
            "estimator__gamma": scipy.stats.uniform(0, 5),
            "estimator__min_child_weight": scipy.stats.randint(1, 10),
        },
    }

class FoldSMOTE(BaseEstimator):
    def __init__(self, random_state=1234, k_neighbors=5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        counts = pd.Series(y).value_counts()
        if len(counts) < 2:
            raise ValueError('Oversampling requires two classes in each training fold.')
        if counts.nunique() == 1:
            return X, y
        if counts.min() < 2:
            raise ValueError('SMOTE needs at least two minority examples in each training fold.')
        k = min(self.k_neighbors, int(counts.min()) - 1)
        categorical = X.select_dtypes(exclude=np.number).columns.tolist()
        kwargs = dict(random_state=self.random_state, k_neighbors=k)
        if not categorical:
            sampler = SMOTE(**kwargs)
        elif len(categorical) == X.shape[1]:
            sampler = SMOTEN(**kwargs)
        else:
            sampler = SMOTENC(categorical_features=categorical, **kwargs)
        return sampler.fit_resample(X, y)

def make_scoring(config):
    safe = {
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
    }
    return {name: safe.get(scorer, scorer) if isinstance(scorer, str) else scorer for name, scorer in config.scoring_dict.items()}

def make_pipeline(estimator, config, feature_engineering=False, smote=False):
    steps = []
    if feature_engineering:
        steps.append(('feature_engineering', DataTransform()))
    if smote:
        prepare = ColumnTransformer([
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
                ('scaler', MinMaxScaler()),
            ]), make_column_selector(dtype_include=np.number)),
            ('categorical', SimpleImputer(strategy='most_frequent', keep_empty_features=True),
             make_column_selector(dtype_exclude=np.number)),
        ], verbose_feature_names_out=False).set_output(transform='pandas')
        steps.append(('sampling_preparation', prepare))
        steps.append(('smote', FoldSMOTE(config.seed, config.smote_k_neighbors)))
    steps.append(('preprocessing', make_preprocessor()))
    steps.append(('estimator', clone(estimator)))
    return Pipeline(steps)

def make_cv(config: Config) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.cross_validation_amount,
        shuffle=True,
        random_state=config.seed,
    )

def run_cross_validation(pipeline, X_train, y_train, config):
    return cross_validate(
        estimator=pipeline, X=X_train, y=y_train,
        scoring=make_scoring(config), cv=make_cv(config),
        return_train_score=True, n_jobs=config.n_jobs,
        verbose=config.verbose, error_score='raise',
    )

def run_random_search(pipeline, parameters, X_train, y_train, config, fit=True):
    search = RandomizedSearchCV(
        estimator=clone(pipeline), param_distributions=parameters,
        n_iter=config.hyperparameter_test_amount, scoring=make_scoring(config),
        cv=make_cv(config), refit=config.refit, n_jobs=1,
        random_state=config.seed, return_train_score=True,
        verbose=config.verbose, error_score='raise',
    )
    if fit:
        search.fit(X_train, y_train)
    return search

class Benchmark:
    def __init__(
        self,
        config: Config,
        data: DataContext,
    ) -> None:
        self.config = config
        self.data = data
        self.results: list[dict[str, Any]] = []
        self.best_models: dict[str, Any] = {}

    def run(self) -> pd.DataFrame:
        self.results.clear()
        self.best_models.clear()
        split_data(self.data, self.config)
        if self.data.y_train.value_counts().min() < self.config.cross_validation_amount and (self.config.use_cv or self.config.use_grid):
            raise ValueError('Too few training examples per class for the configured CV folds.')
        parameters = model_parameters()
        for model_name, estimator in tqdm.tqdm(model_variants(self.config).items(), desc='Benchmark'):
            self.run_model(model_name, estimator, parameters.get(model_name, {}))
        results = pd.DataFrame(self.results)
        if self.config.save_results:
            results.to_csv(self.config.results_path, index=False)
        return results

    def run_model(
        self,
        model_name: str,
        estimator: Any,
        parameters: dict[str, Any],
    ) -> None:
        if self.config.debug:
            print(f"\n{'=' * 60}\nMODEL: {model_name}\n{'=' * 60}")

        base_pipeline = make_pipeline(
            estimator,
            self.config,
            feature_engineering=False,
            smote=self.config.use_smote,
        )

        self.run_experiment(
            model_name,
            "base",
            base_pipeline,
            parameters,
        )

        if self.config.compare_feature_engineering:
            transformed_pipeline = make_pipeline(
                estimator,
                self.config,
                feature_engineering=True,
                smote=self.config.use_smote,
            )

            self.run_experiment(
                model_name,
                "transformed",
                transformed_pipeline,
                parameters,

            )

    def run_experiment(self, model_name, experiment_name, pipeline, parameters):
        X_train, y_train = self.data.X_train, self.data.y_train
        use_search = self.config.use_grid and bool(parameters)
        estimator = run_random_search(pipeline, parameters, None, None, self.config, fit=False) if use_search else clone(pipeline)
        if self.config.use_cv:
            cv = run_cross_validation(estimator, X_train, y_train, self.config)
            for metric in self.config.scoring_dict:
                self.results.append({
                    'model': model_name, 'experiment': experiment_name,
                    'method': 'nested_cross_validation' if use_search else 'cross_validation',
                    'scoring': metric, 'score': cv[f'test_{metric}'].mean(),
                    'score_std': cv[f'test_{metric}'].std(),
                    'train_score': cv[f'train_{metric}'].mean(),
                })
        fitted = clone(estimator).fit(X_train, y_train)
        final_model = fitted.best_estimator_ if use_search else fitted
        key = f'{model_name}_{experiment_name}'
        self.best_models[key] = final_model
        if use_search:
            self.results.append({
                'model': model_name, 'experiment': experiment_name,
                'method': 'inner_random_search', 'scoring': self.config.refit,
                'score': fitted.best_score_, 'score_std': np.nan, 'train_score': np.nan,
                'best_params': json.dumps(fitted.best_params_, default=str),
            })
        test_scores = self.evaluate_test(final_model)
        for name, scorer in make_scoring(self.config).items():
            self.results.append({
                'model': model_name, 'experiment': experiment_name,
                'method': 'holdout_test', 'scoring': name,
                'score': test_scores[name], 'score_std': np.nan,
                'train_score': get_scorer(scorer)(final_model, X_train, y_train) if isinstance(scorer, str) else scorer(final_model, X_train, y_train),
            })

    def evaluate_test(self, model):
        if self.data.X_test is None or self.data.y_test is None:
            raise RuntimeError('Test data is not available.')
        return {
            name: (get_scorer(scorer) if isinstance(scorer, str) else scorer)(model, self.data.X_test, self.data.y_test)
            for name, scorer in make_scoring(self.config).items()
        }

    def save_best_models(self) -> None:
        for name, model in self.best_models.items():
            self.config.save_model(model, name)

def main() -> None:
    config = Config(
        target= "Exited",
        seed=1234,
        test_size=0.20,

        use_cv=True,
        cross_validation_amount=5,

        use_grid=True,
        hyperparameter_test_amount=5,

        scoring_result="f1",
        refit="f1",

        use_smote=True,
        smote_k_neighbors=5,
        compare_feature_engineering=True,

        debug=True,
        verbose=0,
        n_jobs=-1,

        saving_model=False,
        save_results=True,
        use_tkinter=False,
    )

    config.print()

    data = DataLoader(config).load()

    if config.debug:
        print(f"Dataset shape: {data.df.shape}")
        print(f"Numerical: {data.numerical_columns}")
        print(f"Categorical: {data.categorical_columns}")

    if config.open_csv_directory:
        config.open_csv_dir()
    benchmark = Benchmark(config, data)
    results = benchmark.run()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))

    benchmark.save_best_models()

if __name__ == "__main__":
    main()
