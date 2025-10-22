import os
import sys
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_classification_models(X_train, y_train, X_test, y_test, models, param_grid=None):
    try:
        report = {}
        for model_name, model in models.items():
            if param_grid and model_name in param_grid:
                grid = GridSearchCV(model, param_grid[model_name], cv=3)
                grid.fit(X_train, y_train)
                model.set_params(**grid.best_params_)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report[model_name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred)
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)
