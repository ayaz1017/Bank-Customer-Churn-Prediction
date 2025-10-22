import sys
import pandas as pd
from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        try:
            # Load saved preprocessor and model
            self.model = load_object("artifacts/best_model.pkl")
            self.preprocessor = load_object("artifacts/preprocessor.pkl")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame):
        """
        Takes a DataFrame as input, applies preprocessing, and returns predictions.
        """
        try:
            # Transform the input using the saved preprocessor
            data_scaled = self.preprocessor.transform(input_df)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class to convert form inputs into a DataFrame for prediction.
    Update feature names according to your dataset.
    """
    def __init__(
        self,
        CreditScore: float,
        Geography: str,
        Gender: str,
        Age: float,
        Tenure: float,
        Balance: float,
        NumOfProducts: float,
        HasCrCard: float,
        IsActiveMember: float,
        EstimatedSalary: float
    ):
        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary

    def get_data_as_data_frame(self):
        """
        Converts the input data into a DataFrame for prediction.
        """
        try:
            data_dict = {
                "CreditScore": [self.CreditScore],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
                "NumOfProducts": [self.NumOfProducts],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
