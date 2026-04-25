import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Constructs the ColumnTransformer WITHOUT fitting it.
        Separating construction from fitting means we can inspect or test
        the transformer independently of the data it will eventually run on.

        FEATURE DESIGN DECISIONS:

        Numeric pipeline (calories, carbohydrate, sugar, protein, servings):
          - Step 1: Median imputation. We use the MEDIAN (not mean) because
            all four nutrition columns are right-skewed — a small number of
            very high-calorie recipes pull the mean upward, making it a poor
            representative value for filling missing rows.
          - Step 2: StandardScaler (zero mean, unit variance). Without scaling,
            calories (range 0–3000+) would numerically dominate protein (range
            0–50) in any distance-based or gradient-based model, purely because
            of magnitude differences — not because calories are actually more
            predictive.

        Categorical pipeline (category):
          - Step 1: Most-frequent imputation (defensive; no missing values
            expected after cleaning, but good practice).
          - Step 2: OneHotEncoder with handle_unknown='ignore'. This means if
            a category the model never saw during training appears at inference
            time, it produces an all-zero row rather than raising an error.
            sparse_output=False makes it easier to concatenate with the numeric
            features inside the ColumnTransformer.
          - Step 3: StandardScaler with with_mean=False. Setting with_mean=False
            is necessary when dealing with sparse-like binary output — though our
            OHE is not sparse (sparse_output=False), keeping this setting is a
            safe default and puts the binary columns on the same numeric scale
            as the processed numeric features, which benefits linear models.
        """
        try:
            numerical_columns = [
                "calories", "carbohydrate", "sugar", "protein", "servings"
            ]
            categorical_columns = ["category"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Loads train and test CSVs, fits the preprocessor on training data
        ONLY (to prevent data leakage), transforms both splits, and saves
        the fitted preprocessor to disk.

        WHY fit_transform() ON TRAIN BUT transform() ON TEST?
        If we called fit_transform() on the test set, the scaler would
        learn the test set's mean and standard deviation. That means the
        scaling parameters used for test evaluation would be informed by
        the test data — a subtle form of data leakage that produces
        optimistically biased metrics. By fitting only on train and then
        calling transform() on test, we guarantee the test set is treated
        as completely unseen data at all times.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "high_traffic"
            feature_columns = [
                "calories", "carbohydrate", "sugar", "protein", "category", "servings"
            ]

            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing dataframes."
            )

            # CRITICAL: fit only on training data, then transform both sets.
            input_feature_train_array = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_array = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Concatenate features and target back into single arrays so the
            # model trainer receives everything in one clean package.
            train_arr = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            # Persist the FITTED preprocessor. The inference pipeline loads
            # this at prediction time to apply the exact same imputation
            # medians, scaler parameters, and OHE vocabulary as training.
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Saved preprocessing object to artifacts/preprocessor.pkl.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)