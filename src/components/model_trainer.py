import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_classification_models, save_object

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting Model Trainer")

            # Define models to evaluate
            models = {
                "Logistic Regression": LogisticRegression(),
                "SVC": SVC(),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": xgb.XGBClassifier(eval_metric='logloss')  # Removed use_label_encoder warning
            }

            # Evaluate models
            scores = evaluate_classification_models(X_train, y_train, X_test, y_test, models)

            # Identify the best model
            best_model_name = max(scores, key=lambda k: scores[k]["Accuracy"])
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with accuracy: {scores[best_model_name]['Accuracy']:.4f}")

            # Optionally: save the best model
            save_object(file_path='artifacts/model.pkl', obj=best_model)
            logging.info("Best model saved at artifacts/model.pkl")

            return best_model_name, scores, best_model

        except Exception as e:
            raise CustomException(e, sys)
