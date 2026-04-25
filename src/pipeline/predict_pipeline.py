import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor artifacts.")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # transform() — not fit_transform() — see class docstring above.
            data_scaled = preprocessor.transform(features)

            predictions = model.predict(data_scaled)
            # predict_proba() returns [P(Not High), P(High)]; we want column 1.
            probabilities = model.predict_proba(data_scaled)[:, 1]

            return predictions, probabilities

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    VALID_CATEGORIES = [
        "Chicken", "Beverages", "Breakfast", "Dessert",
        "Lunch/Snacks", "Meat", "One Dish Meal", "Pork",
        "Potato", "Vegetable",
    ]

    def __init__(
        self,
        calories: float,
        carbohydrate: float,
        sugar: float,
        protein: float,
        category: str,
        servings: int,
    ):
        self.calories = calories
        self.carbohydrate = carbohydrate
        self.sugar = sugar
        self.protein = protein
        self.category = category
        self.servings = servings

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "calories":     [self.calories],
                "carbohydrate": [self.carbohydrate],
                "sugar":        [self.sugar],
                "protein":      [self.protein],
                "category":     [self.category],
                "servings":     [self.servings],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)