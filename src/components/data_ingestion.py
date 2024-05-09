from src.exception import CustomException
from src.logger import logging

import pandas as pd 
import numpy as np 
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    row_data_path:str=os.path.join('artifacts','row.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initial_data_ingestion(self):
        logging.info('Enter the data ingestion method or component')

        try:
            df=pd.read_csv('notebook\data\students.csv')
            logging.info('Read the dataset as data frame compelet')

            os.makedirs(os.path.dirname(self.ingestion_config.row_data_path),exist_ok=True)


            df.to_csv(self.ingestion_config.row_data_path,header=True,index=False)

            logging.info('Train_test_split initiated')

            train_set,test_set=train_test_split(df,random_state=42,test_size=0.2)

            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        

        except Exception as e:
            raise CustomException(sys,e)
        
if __name__=='__main__':
    obj=DataIngestion()
    obj.initial_data_ingestion()

            