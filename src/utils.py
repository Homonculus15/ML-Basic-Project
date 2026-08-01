import os
import sys

import numpy as np 
import pandas as pd 
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        report={}

        for i in range(len(list(model))):
            model_name = list(model.keys())[i]
            model_obj = list(model.values())[i]
            
            model_obj.fit(X_train, y_train)
            
            y_train_pred=model_obj.predict(X_train)
            y_test_pred=model_obj.predict(X_test)
            
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            
            report[model_name]=test_model_score
            
        return report

            
    except Exception as e:
        raise CustomException(e,sys)
        
