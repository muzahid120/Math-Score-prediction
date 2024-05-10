import os 
import sys
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessore_file_obj:str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config=DataTransformationConfig()

    def get_transformation_obj(self):

        try:
            numerical_features= ["writing_score", "reading_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]


            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaled',StandardScaler())
            ])

            logging.info('Numerical  pipeline completed')

            cat_pipeline=Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent'))
            ,('One Hot Encoding',OneHotEncoder()),
            ('scaled',StandardScaler(with_mean=False))]
    )
            logging.info('Categorical pipeline compeleted')

            preprocessor=ColumnTransformer([('num_pipeline',num_pipeline,numerical_features),
                                            ('cat_pipeline',cat_pipeline,categorical_features)])
            
            logging.info('Preprocessing is completed')

            return preprocessor



        except Exception as e:
            raise CustomException(sys,e)


    def initial_data_transformation(self,train_data_path,test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            logging.info('Train data set reaing successfull')
            test_df=pd.read_csv(test_data_path)
            logging.info('Test data set reaing successfull')

            target_column_name="math_score"

            input_train_features=train_df.drop(columns=[target_column_name],axis=1)
            target_train_feature=train_df[target_column_name]


            input_test_features=test_df.drop(columns=[target_column_name],axis=1)
            target_test_features=test_df[target_column_name]


            preprocessore_obj= self.get_transformation_obj()
            input_train_arr=preprocessore_obj.fit_transform(input_train_features)
            input_test_arr=preprocessore_obj.transform(input_test_features)


            train_arr=np.c_[input_train_arr,np.array(target_train_feature)]
            test_arr=np.c_[input_test_arr,np.array(target_test_features)]

            save_object(file_path=self.data_transfromation_config.preprocessore_file_obj,
                        obj=preprocessore_obj)
            

            return(train_arr,
                   test_arr,
                   self.data_transfromation_config.preprocessore_file_obj)

            


        except Exception as e:
            raise CustomException(sys,e)