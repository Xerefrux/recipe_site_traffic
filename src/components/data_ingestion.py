import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Centralises every file-path decision for Stage 1.
    Using os.path.join() throughout (rather than hardcoded slashes) ensures
    the project runs on both Windows (development) and Linux (Render/AWS).
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all documented data cleaning decisions to the raw dataframe.
        Keeping this in its own method makes the logic auditable independently
        of the surrounding file I/O.

        DECISIONS DOCUMENTED:
        1. Drop 'recipe' — it is a unique identifier, not a predictive feature.
        2. Merge 'Chicken Breast' → 'Chicken' — the data dictionary specifies
           exactly 10 categories; 'Chicken Breast' is an undocumented 11th that
           clearly belongs under 'Chicken'.
        3. Extract leading integer from servings strings like '4 as a snack'
           — the snack context is already captured by the category column.
        4. Encode target: 'High' → 1, anything else (NaN, blank) → 0.
        5. Leave the 52 rows with all-NaN nutrition values as NaN — the
           SimpleImputer in Stage 2 will fill them with column medians, which
           is the statistically correct approach for right-skewed distributions.
        """
        logging.info("Starting raw data cleaning.")

        # Decision 1: drop the ID column
        df = df.drop(columns=["recipe"])

        # Decision 2: consolidate the undocumented 'Chicken Breast' category
        df["category"] = df["category"].replace("Chicken Breast", "Chicken")
        logging.info(
            f"After merging 'Chicken Breast' into 'Chicken': "
            f"{df['category'].nunique()} unique categories."
        )

        # Decision 3: extract integer from strings like '4 as a snack'
        df["servings"] = (
            df["servings"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(int)
        )

        # Decision 4: binary-encode the target variable
        df["high_traffic"] = df["high_traffic"].apply(
            lambda x: 1 if str(x).strip() == "High" else 0
        )
        pct_high = df["high_traffic"].mean() * 100
        logging.info(
            f"Target encoded. Class balance — High: {pct_high:.1f}% | "
            f"Not High: {100 - pct_high:.1f}%"
        )

        logging.info("Raw data cleaning complete.")
        return df

    def initiate_data_ingestion(self):
        """
        Reads the raw CSV, cleans it, performs a stratified 80/20 split,
        and persists all three versions to artifacts/.

        WHY STRATIFIED SPLIT?
        The target class ratio is approximately 60% High / 40% Not High.
        A plain random split could, by chance, produce a test set with a
        very different ratio, making our evaluation metrics misleading.
        stratify=df["high_traffic"] guarantees the ratio is preserved in
        both train and test sets.
        """
        logging.info("Entered the data ingestion method.")
        try:
            df = pd.read_csv(os.path.join("data", "recipe_site_traffic_2212.csv"))
            logging.info(
                f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns."
            )

            df = self._clean_raw_data(df)

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train/test split initiated.")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["high_traffic"],  # preserve class balance
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info(
                f"Ingestion complete — train: {len(train_set)} rows, "
                f"test: {len(test_set)} rows."
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Running this file directly kicks off the full 3-stage training pipeline.
    # Each stage imports and calls the next, so a single command triggers
    # Stage 1 → Stage 2 → Stage 3 in sequence.
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))