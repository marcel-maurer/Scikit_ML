import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

from pathlib import Path
from textwrap import dedent

class PlotManager():

    def __init__(self):
        self.base_vs_transformed_plot = None

    def dependencies(self):
        try:
            import matplotlib.pyplot as plt

        except Exception as import_error:
            print(f"imp error:{import_error} ")
            return


    
    def plot_scores(
        data,
        x_column,
        score_columns,
        title="Model Comparison",
        ylabel="Score"
    ):
        # Falls eine Liste von Dictionaries übergeben wird
        if isinstance(data, list):
            data = pd.DataFrame(data)

        x = range(len(data))

        plt.figure(figsize=(12, 6))

        for column in score_columns:
            if column in data.columns:
                plt.plot(
                    x,
                    data[column],
                    marker="o",
                    linewidth=2,
                    label=column
                )

        plt.xticks(x, data[x_column], rotation=45)
        plt.xlabel(x_column)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def v2(self , data_frame):
        df = data_frame.DataFrame(self.results)

        x = np.arange(len(df))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.errorbar(
            x - width / 2,
            df["Base_test_score"],
            yerr=df["Base_test_score"],
            fmt="o-",
            capsize=5,
            linewidth=2,
            label="Ohne Transformation"
        )

        ax.errorbar(
            x + width / 2,
            df["Transformed_test_score"],
            yerr=df["Transformed_test_score"],
            fmt="s-",
            capsize=5,
            linewidth=2,
            label="Mit Transformation"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [type(est).__name__ for est in df["Estimator"]],
            rotation=30,
            ha="right"
        )

        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Modelle")
        ax.set_title("Modellvergleich (Cross Validation)")
        ax.set_ylim(0, 1)

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()

        plt.tight_layout()
        plt.show()


    def grouped_bar_plot(self, df):

        x = np.arange(len(df))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.bar(
            x - width,
            df["roc_score"],
            width,
            label="Base ROC-AUC"
        )

        ax.bar(
            x,
            df["tuned_roc_score"],
            width,
            label="Tuned ROC-AUC"
        )

        ax.bar(
            x + width,
            df["transformed_roc_score"],
            width,
            label="Transformed ROC-AUC"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            df["estimator"],
            rotation=45,
            ha="right"
        )

        ax.set_ylabel("ROC-AUC")
        ax.set_xlabel("Model")
        ax.set_title("ROC-AUC Comparison")

        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        ax.legend()

        plt.tight_layout()
        plt.show()

    def performance_heatmap(self, df):

        metrics = [
            "roc_score",
            "f1_score",
            "pr_score",
            "tuned_roc_score",
            "tuned_f1_score",
            "tuned_pr_score"
        ]

        heatmap_data = df.set_index("estimator")[metrics]

        plt.figure(figsize=(12, 8))

        plt.imshow(
            heatmap_data,
            aspect="auto"
        )

        plt.xticks(
            range(len(metrics)),
            metrics,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(heatmap_data)),
            heatmap_data.index
        )

        plt.colorbar(label="Score")

        plt.title("Model Performance Heatmap")

        plt.tight_layout()
        plt.show()

    def improvement_plot(self, df):

        df = df.copy()

        df["roc_improvement"] = (
            df["tuned_roc_score"]
            - df["roc_score"]
        )

        x = np.arange(len(df))

        plt.figure(figsize=(12, 6))

        plt.bar(
            x,
            df["roc_improvement"]
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xticks(
            x,
            df["estimator"],
            rotation=45,
            ha="right"
        )

        plt.ylabel("ROC-AUC Improvement")
        plt.xlabel("Model")
        plt.title("Improvement After Hyperparameter Tuning")

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.5
        )

        plt.tight_layout()
        plt.show()

    def radar_plot(self, df, model_name):

        row = df[
            df["estimator"] == model_name
        ].iloc[0]

        metrics = [
            "roc_score",
            "f1_score",
            "pr_score"
        ]

        values = [
            row["roc_score"],
            row["f1_score"],
            row["pr_score"]
        ]

        values += values[:1]

        angles = np.linspace(
            0,
            2 * np.pi,
            len(metrics) + 1
        )

        fig = plt.figure(figsize=(7, 7))

        ax = fig.add_subplot(
            111,
            polar=True
        )

        ax.plot(
            angles,
            values,
            marker="o"
        )

        ax.fill(
            angles,
            values,
            alpha=0.2
        )

        ax.set_xticks(
            angles[:-1]
        )

        ax.set_xticklabels(
            metrics
        )

        ax.set_ylim(0, 1)

        ax.set_title(
            f"Performance Profile: {model_name}"
        )

        plt.show()

    def base_vs_tuned_scatter(self, df):

        plt.figure(figsize=(8, 8))

        plt.scatter(
            df["roc_score"],
            df["tuned_roc_score"],
            s=100
        )

        min_value = min(
            df["roc_score"].min(),
            df["tuned_roc_score"].min()
        )

        max_value = max(
            df["roc_score"].max(),
            df["tuned_roc_score"].max()
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        for _, row in df.iterrows():

            plt.annotate(
                row["estimator"],
                (
                    row["roc_score"],
                    row["tuned_roc_score"]
                )
            )

        plt.xlabel("Base ROC-AUC")
        plt.ylabel("Tuned ROC-AUC")

        plt.title(
            "Base vs Tuned Model Performance"
        )

        plt.grid(
            linestyle="--",
            alpha=0.5
        )

        plt.tight_layout()
        plt.show()

    def model_metric_plot(self, df):

        metrics = [
            "roc_score",
            "f1_score",
            "pr_score"
        ]

        x = np.arange(len(metrics))

        for _, row in df.iterrows():

            values = [
                row["roc_score"],
                row["f1_score"],
                row["pr_score"]
            ]

            plt.figure(figsize=(8, 5))

            plt.bar(
                x,
                values
            )

            plt.xticks(
                x,
                metrics
            )

            plt.ylim(0, 1)

            plt.ylabel("Score")

            plt.title(
                f"{row['estimator']} Performance"
            )

            plt.grid(
                axis="y",
                linestyle="--",
                alpha=0.5
            )

            plt.tight_layout()

            plt.show()
            
class ParamGrid():

    def __init__(self):
        self.available_estimators = self.count()
        self.count()

    def scoring_dict(self):

        return {

            "f1":"f1",
            "accuracy":"accuracy",
            "precision":"precision",

        }

    def custom_paramter_grid(self):

        return {

            "LogisticRegression": {
                "estimator__C": scipy.stats.loguniform(1e-4, 1e2),
                "estimator__penalty": ["l2"],
                "estimator__solver": ["lbfgs"],
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

            "LinearSVC": {
                "estimator__C": scipy.stats.loguniform(1e-4, 1e2),
                "estimator__loss": ["hinge", "squared_hinge"],
                "estimator__dual": [True],
                "estimator__max_iter": [5000, 10000],
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





class MLCodeGenerator:

    def __init__(self):

        self.project_name = "generated_ml_project.py"

        self.options = {
            "config": True,
            "data_class": True,
            "data_transform": True,
            "preprocessor": True,
            "models": True,
            "pipeline": True,
            "scoring": True,
            "parameters": True,
            "cross_validation": True,
            "grid_search": True,
            "data_split": True,
            "benchmark": True,
        }


    # ============================================================
    # IMPORTS
    # ============================================================

    IMPORTS = dedent("""
        import numpy as np
        import pandas as pd
        import scipy

        from dataclasses import dataclass

        from sklearn.base import BaseEstimator, TransformerMixin

        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier

        from xgboost import XGBClassifier

        from sklearn.model_selection import (
            train_test_split,
            cross_validate,
            KFold,
            RandomizedSearchCV
        )

        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            MinMaxScaler,
            OneHotEncoder
        )
    """)


    # ============================================================
    # CONFIG
    # ============================================================

    CONFIG = dedent("""
        @dataclass
        class Config:

            target: str = "Survived"

            seed: int = 1234

            test_size: float = 0.2

            hyperparameter_test_amount: int = 5

            cross_validation_amount: int = 5

            verbose: int = 0


        config = Config()
    """)

    # ============================================================
    # Get CSV Path
    # ============================================================
    
    def get_csv_path():
        csv_path = input("Enter Path to CSV")
        return csv_path

    # ============================================================
    # DATA LOADER
    # ============================================================
    
    DATA_LOADER = dedent("""
        # ============================================================
        # DATA LOADING
        # ============================================================

        linux_path = r""
        windows_path = r"I:/path/to/windows/dataset.csv"

        df = pd.read_csv(linux_path)
    """)


    # ============================================================
    # DATA CLASS
    # ============================================================

    DATA_CLASS = dedent("""
        # ============================================================
        # DATA CLASS
        # ============================================================

        class DataClass:

            def __init__(self, DataFrame):

                self.data = DataFrame.copy()

                self.x = self.data.drop(
                    columns=[config.target]
                )

                self.y = self.data[config.target]

                self.numerical_data = (
                    self.x
                    .select_dtypes(include=np.number)
                    .columns
                    .tolist()
                )

                self.categorical_data = (
                    self.x
                    .select_dtypes(exclude=np.number)
                    .columns
                    .tolist()
                )


        data = DataClass(DataFrame=df)
    """)


    # ============================================================
    # DATA TRANSFORM
    # ============================================================

    DATA_TRANSFORM = dedent("""
        # ============================================================
        # DATA TRANSFORMATION
        # ============================================================

        class DataTransform(
            BaseEstimator,
            TransformerMixin
        ):

            def fit(
                self,
                X,
                y=None
            ):

                return self


            def transform(
                self,
                X
            ):

                x = X.copy()

                x = self.new_features(x)

                return x


            def new_features(
                self,
                X
            ):

                x = X.copy()

                # ====================================================
                # ADD YOUR CUSTOM FEATURE ENGINEERING HERE
                # ====================================================

                return x
    """)


    # ============================================================
    # PREPROCESSOR
    # ============================================================

    PREPROCESSOR = dedent("""
        # ============================================================
        # PREPROCESSOR
        # ============================================================

        class Preprocessor(
            BaseEstimator,
            TransformerMixin
        ):

            def fit(
                self,
                X,
                y=None
            ):

                numerical_columns = (
                    X.select_dtypes(
                        include=np.number
                    ).columns
                )

                categorical_columns = (
                    X.select_dtypes(
                        exclude=np.number
                    ).columns
                )


                self.preprocessor = ColumnTransformer(

                    transformers=[

                        (
                            "numerical_handler",

                            Pipeline([

                                (
                                    "numerical_imputer",

                                    SimpleImputer(
                                        strategy="mean"
                                    )
                                ),

                                (
                                    "scaler",

                                    MinMaxScaler()
                                )
                            ]),

                            numerical_columns
                        ),


                        (
                            "categorical_handler",

                            Pipeline([

                                (
                                    "categorical_imputer",

                                    SimpleImputer(
                                        strategy="most_frequent"
                                    )
                                ),

                                (
                                    "categorical_encoder",

                                    OneHotEncoder(
                                        handle_unknown="ignore",
                                        sparse_output=False
                                    )
                                )
                            ]),

                            categorical_columns
                        )
                    ]
                )


                self.preprocessor.fit(
                    X,
                    y
                )

                return self


            def transform(
                self,
                X
            ):

                return self.preprocessor.transform(
                    X
                )
    """)


    # ============================================================
    # MODELS
    # ============================================================

    MODEL_VARIANCE = dedent("""
        # ============================================================
        # MODEL VARIANCE
        # ============================================================

        def model_variance():

            return {

                "LogisticRegression":

                    LogisticRegression(
                        random_state=config.seed
                    ),


                "LinearSVC":

                    LinearSVC(
                        random_state=config.seed
                    ),


                "DecisionTreeClassifier":

                    DecisionTreeClassifier(
                        random_state=config.seed
                    ),


                "RandomForestClassifier":

                    RandomForestClassifier(
                        random_state=config.seed
                    ),


                "XGBClassifier":

                    XGBClassifier(
                        random_state=config.seed
                    )
            }
    """)


    # ============================================================
    # PIPELINE
    # ============================================================

    PIPELINE = dedent("""
        # ============================================================
        # PIPELINE
        # ============================================================

        def custom_pipeline(
            estimator,
            transform=False
        ):

            steps = []


            if transform:

                steps.append(
                    (
                        "data_transform",
                        DataTransform()
                    )
                )


            steps.append(
                (
                    "preprocessor",
                    Preprocessor()
                )
            )


            steps.append(
                (
                    "estimator",
                    estimator
                )
            )


            return Pipeline(
                steps
            )
    """)


    # ============================================================
    # SCORING
    # ============================================================

    SCORING = dedent("""
        # ============================================================
        # SCORING
        # ============================================================

        def custom_scoring():

            return {

                "accuracy":
                    "accuracy",

                "f1":
                    "f1",

                "precision":
                    "precision",

                "recall":
                    "recall"
            }
    """)


    # ============================================================
    # PARAMETERS
    # ============================================================

    PARAMETERS = dedent("""
        # ============================================================
        # HYPERPARAMETERS
        # ============================================================

        def custom_params():

            return {

                "LogisticRegression": {

                    "estimator__C":
                        scipy.stats.loguniform(
                            1e-4,
                            1e2
                        ),

                    "estimator__penalty":
                        ["l2"],

                    "estimator__solver":
                        ["lbfgs"]
                },


                "DecisionTreeClassifier": {

                    "estimator__criterion":
                        ["gini", "entropy"],

                    "estimator__max_depth":
                        [None, 5, 10, 20],

                    "estimator__min_samples_split":
                        scipy.stats.randint(
                            2,
                            20
                        )
                },


                "RandomForestClassifier": {

                    "estimator__n_estimators":
                        scipy.stats.randint(
                            100,
                            500
                        ),

                    "estimator__max_depth":
                        scipy.stats.randint(
                            5,
                            40
                        ),

                    "estimator__min_samples_split":
                        scipy.stats.randint(
                            2,
                            20
                        )
                },


                "LinearSVC": {

                    "estimator__C":
                        scipy.stats.loguniform(
                            1e-4,
                            1e2
                        )
                },


                "XGBClassifier": {

                    "estimator__n_estimators":
                        scipy.stats.randint(
                            100,
                            500
                        ),

                    "estimator__learning_rate":
                        scipy.stats.loguniform(
                            1e-3,
                            0.3
                        ),

                    "estimator__max_depth":
                        scipy.stats.randint(
                            3,
                            10
                        )
                }
            }
    """)


    # ============================================================
    # CROSS VALIDATION
    # ============================================================

    CROSS_VALIDATION = dedent("""
        # ============================================================
        # CROSS VALIDATION
        # ============================================================

        def custom_cross_validation(
            estimator,
            X_train,
            y_train
        ):

            k_fold = KFold(

                n_splits=
                    config.cross_validation_amount,

                shuffle=True,

                random_state=
                    config.seed
            )


            return cross_validate(

                estimator=estimator,

                X=X_train,

                y=y_train,

                cv=k_fold,

                n_jobs=-1,

                return_train_score=True,

                scoring=custom_scoring(),

                verbose=config.verbose
            )
    """)


    # ============================================================
    # RANDOMIZED SEARCH
    # ============================================================

    GRID_SEARCH = dedent("""
        # ============================================================
        # RANDOMIZED SEARCH
        # ============================================================

        def custom_grid_search(
            estimator,
            params
        ):

            return RandomizedSearchCV(

                estimator=estimator,

                param_distributions=params,

                n_iter=
                    config.hyperparameter_test_amount,

                cv=
                    config.cross_validation_amount,

                n_jobs=-1,

                scoring=custom_scoring(),

                refit="accuracy",

                return_train_score=True
            )
    """)


    # ============================================================
    # DATA SPLIT
    # ============================================================

    DATA_SPLIT = dedent("""
        # ============================================================
        # DATA SPLIT
        # ============================================================

        def data_split():

            X_train, X_test, y_train, y_test = (

                train_test_split(

                    data.x,

                    data.y,

                    test_size=
                        config.test_size,

                    random_state=
                        config.seed,

                    shuffle=True,

                    stratify=data.y
                )
            )


            print(
                f"X_train shape: "
                f"{X_train.shape}"
            )

            print(
                f"X_test shape: "
                f"{X_test.shape}"
            )

            print(
                f"y_train shape: "
                f"{y_train.shape}"
            )

            print(
                f"y_test shape: "
                f"{y_test.shape}"
            )


            return (

                X_train,

                X_test,

                y_train,

                y_test
            )
    """)


    # ============================================================
    # BENCHMARK
    # ============================================================

    BENCHMARK = dedent("""
        # ============================================================
        # BENCHMARK
        # ============================================================

        class Benchmark:

            def __init__(self):

                (
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test

                ) = data_split()


                self.results = []


            def fit(self):

                for (

                    estimator_name,

                    estimator

                ) in model_variance().items():


                    print(
                        f"Running: "
                        f"{estimator_name}"
                    )


                    base_pipeline = (

                        custom_pipeline(

                            estimator=estimator,

                            transform=False
                        )
                    )


                    transformed_pipeline = (

                        custom_pipeline(

                            estimator=estimator,

                            transform=True
                        )
                    )


                    # ============================================
                    # BASE CV
                    # ============================================

                    base_cv = (

                        custom_cross_validation(

                            estimator=base_pipeline,

                            X_train=self.X_train,

                            y_train=self.y_train
                        )
                    )


                    # ============================================
                    # TRANSFORMED CV
                    # ============================================

                    transformed_cv = (

                        custom_cross_validation(

                            estimator=
                                transformed_pipeline,

                            X_train=self.X_train,

                            y_train=self.y_train
                        )
                    )


                    self.results.append({

                        "Estimator":
                            estimator_name,


                        "Base_CV_Train_Accuracy":

                            base_cv[
                                "train_accuracy"
                            ].mean(),


                        "Base_CV_Test_Accuracy":

                            base_cv[
                                "test_accuracy"
                            ].mean(),


                        "Base_CV_STD":

                            base_cv[
                                "test_accuracy"
                            ].std(),


                        "Transformed_CV_Train_Accuracy":

                            transformed_cv[
                                "train_accuracy"
                            ].mean(),


                        "Transformed_CV_Test_Accuracy":

                            transformed_cv[
                                "test_accuracy"
                            ].mean(),


                        "Transformed_CV_STD":

                            transformed_cv[
                                "test_accuracy"
                            ].std()
                    })


                return self.results
    """)


    # ============================================================
    # GENERATE PROJECT
    # ============================================================

    def generate_project(self):

        code = []


        # Imports are always required

        code.append(
            self.IMPORTS
        )


        if self.options["config"]:

            code.append(
                self.CONFIG
            )


        code.append(
            self.DATA_LOADER
        )


        if self.options["data_class"]:

            code.append(
                self.DATA_CLASS
            )


        if self.options["data_transform"]:

            code.append(
                self.DATA_TRANSFORM
            )


        if self.options["preprocessor"]:

            code.append(
                self.PREPROCESSOR
            )


        if self.options["models"]:

            code.append(
                self.MODEL_VARIANCE
            )


        if self.options["pipeline"]:

            code.append(
                self.PIPELINE
            )


        if self.options["scoring"]:

            code.append(
                self.SCORING
            )


        if self.options["parameters"]:

            code.append(
                self.PARAMETERS
            )


        if self.options["cross_validation"]:

            code.append(
                self.CROSS_VALIDATION
            )


        if self.options["grid_search"]:

            code.append(
                self.GRID_SEARCH
            )


        if self.options["data_split"]:

            code.append(
                self.DATA_SPLIT
            )


        if self.options["benchmark"]:

            code.append(
                self.BENCHMARK
            )


        # ========================================================
        # MAIN
        # ========================================================

        code.append(
            dedent("""
                # ========================================================
                # RUN
                # ========================================================

                if __name__ == "__main__":

                    benchmark = Benchmark()

                    results = benchmark.fit()

                    results_df = pd.DataFrame(
                        results
                    )

                    print(
                        results_df
                    )
            """)
        )


        return "\n\n".join(
            code
        )


    # ============================================================
    # SAVE PROJECT
    # ============================================================

    def save_project(
        self,
        code,
        filename=None
    ):

        if filename is None:

            filename = self.project_name


        path = Path(
            filename
        )


        path.write_text(

            code,

            encoding="utf-8"
        )


        print(
            "\nProject successfully generated!"
        )


        print(
            f"Location: "
            f"{path.absolute()}"
        )


    # ============================================================
    # INTERACTIVE OPTIONS
    # ============================================================

    def configure(self):

        print(
            "\n=========================================="
        )

        print(
            "       MACHINE LEARNING CODE GENERATOR"
        )

        print(
            "==========================================\n"
        )


        print(
            "Select the components "
            "you want to generate.\n"
        )


        for option in self.options:

            current_value = (
                self.options[option]
            )


            answer = input(

                f"Generate {option}? "
                f"[Y/n]: "

            ).strip().lower()


            if answer == "":

                self.options[option] = (
                    current_value
                )

            elif answer in (
                "y",
                "yes"
            ):

                self.options[option] = True

            elif answer in (
                "n",
                "no"
            ):

                self.options[option] = False


    # ============================================================
    # RUN GENERATOR
    # ============================================================

    def run(self):

        self.configure()


        project_name = input(

            "\nEnter output filename "
            "[generated_ml_project.py]: "

        ).strip()


        if project_name:

            if not project_name.endswith(
                ".py"
            ):

                project_name += ".py"


            self.project_name = (
                project_name
            )


        code = (
            self.generate_project()
        )


        self.save_project(
            code
        )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    generator = MLCodeGenerator()

    generator.run()
