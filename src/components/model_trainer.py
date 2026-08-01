import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import( AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
            try:
                logging.info("Split training and test input data")
                X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
                )
                
                models = {
                            "Random Forest": RandomForestRegressor(),
                            "Decision Tree": DecisionTreeRegressor(),
                            "Gradient Boosting": GradientBoostingRegressor(),
                            "Linear Regression": LinearRegression(),
                            "KNeighbors Regressor": KNeighborsRegressor(),
                            "XGBoost Regressor": XGBRegressor(),
                            "CatBoost Regressor": CatBoostRegressor(verbose=False),
                            "AdaBoost Regressor": AdaBoostRegressor()
                        }
                
                params = {
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },

    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },

    "Gradient Boosting": {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200],
        "max_depth": [3, 5]
    },

    "Linear Regression": {},

    "KNeighbors Regressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },

    "XGBoost Regressor": {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7]
    },

    "CatBoost Regressor": {
        "iterations": [100, 500],
        "learning_rate": [0.03, 0.1],
        "depth": [4, 6, 8]
    },

    "AdaBoost Regressor": {
        "learning_rate": [0.01, 0.1, 1],
        "n_estimators": [50, 100, 200]
    }
}
                model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model=models,param=params)
                
                best_model_score=max(sorted(model_report.values()))
                best_model_name=list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
                
                best_model=models[best_model_name]
                best_model.fit(X_train, y_train)
                
                if best_model_score<0.6:
                    raise CustomException("No best model found", sys)
                
                logging.info(f"Best found model on both training and testing dataset")
                
                save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
                
                predicted=best_model.predict(X_test)
                r2=r2_score(y_test,predicted)
                
                return r2
            
            except Exception as e:
                raise CustomException(e,sys)
          