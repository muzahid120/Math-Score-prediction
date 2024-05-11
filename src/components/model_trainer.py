import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evalute_model


@dataclass
class ModelTrainerConfig:
    model_trainer_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('spliting the Train and Test data input')

            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            
            )



            models={'random_forest':RandomForestRegressor(),
                    'xgboost_reg':XGBRegressor(),
                    'adaboost':AdaBoostRegressor(),
                    'linearRegressor':LinearRegression(),
                    'DecisionTreeReg':DecisionTreeRegressor(),
                    'KNN Negiboures':KNeighborsRegressor(),
                    'catboost':CatBoostRegressor(),
                    'GradientBoostingRegressor':GradientBoostingRegressor()}
            
            model_report:dict=evalute_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('NO best Model found')
            logging.info('Best found model on the train and test dataset')

            save_object(self.model_trainer_config.model_trainer_file_path,
                        obj=best_model)
            
            predicted=best_model.predict(x_test)
            R2_score=r2_score(y_test,predicted)

            return R2_score


        
        
        
        
        
        except Exception as e:

            raise CustomException(sys,e)
        

           
    