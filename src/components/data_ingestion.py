import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    best_model_path: str = os.path.join('artifacts', "best_model.pkl")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv(r'notebook/data/bank.csv')  # Update CSV path
            logging.info("Read dataset into dataframe")

            # Drop unnecessary columns
            df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print("Data ingestion completed successfully.")

        # Step 2: Data Transformation
        transformer = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        print(f"Data transformation completed. Preprocessor saved at {preprocessor_path}")

        # Step 3: Model Training
        trainer = ModelTrainer()
        best_model_name, scores, best_model = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

        # Save best model
        save_object(ingestion.ingestion_config.best_model_path, best_model)
        print(f"Best Model: {best_model_name} saved at {ingestion.ingestion_config.best_model_path}")

        print("All model scores:")
        for model, metrics in scores.items():
            print(f"{model}: {metrics}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise CustomException(e, sys)
