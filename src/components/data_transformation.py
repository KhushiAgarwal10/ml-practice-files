import os
import sys
from src.logger import logging
from src.exception import customException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.utils import save_object
@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path=os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransformationConfig()
  
  def get_data_tranformer_obj(self):
    try:
      numerocal_column=['reading score','writing score']
      categorical_column=['gender','race/ethnicity','parental level of education','lunch','test preparation course']

      numerical_pipeline=Pipeline(
        steps=[
          ('imputer',SimpleImputer(strategy='median')),
          ("scaler",StandardScaler())
        ]
      )
      categorical_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy='most_frequent')),
          ("onehotencoder",OneHotEncoder()),
          ("scaler",StandardScaler(with_mean=False))
        ]
      )
      logging.info(f"numerical columns scaling completed: {numerocal_column}")
      logging.info(f"categorical columns encoding completed: {categorical_column}")

      prepocessor=ColumnTransformer([
        ("num_pipeline",numerical_pipeline,numerocal_column),
        ("cat_pipeline",categorical_pipeline,categorical_column)
      ]
      )

      return prepocessor
    except Exception as e:
      raise customException(e,sys)
    
  def initiate_data_tranformation(self,train_path,test_path):
    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)

      logging.info("read train and test data completed")
      logging.info("obtaining preprocessing obj")

      preprocessing_obj=self.get_data_tranformer_obj()
      target_column='math score'
      numerocal_column=['reading score','writing score']

      input_feature_train_df=train_df.drop(columns=[target_column])
      target_feature_train_df=train_df[target_column]

      input_feature_test_df=test_df.drop(columns=[target_column])
      target_feature_test_df=test_df[target_column]


      logging.info(f"applying preprocessing object on training dataframe and testing datafrane ")

      input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

      train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
      test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

      logging.info(f"saved preprocessing object")

      save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessing_obj
      )

      return(
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path,
      )
    except Exception as e:
      raise customException(e,sys)