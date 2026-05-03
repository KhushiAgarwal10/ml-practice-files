import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from src.utils import save_object,evaluate_model
from sklearn.ensemble import(AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import customException
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


@dataclass
class ModelTrainingConfig:
  trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainingConfig()

  def initiate_model_trainer(self,train_arary,test_array):
    try:
      logging.info("split training and test input data")
      x_train,y_train,x_test,y_test=(
        train_arary[:,:-1],train_arary[:,-1],test_array[:,:-1],test_array[:,-1]
      )
      models={
        "lR": LinearRegression(),
        "DT": DecisionTreeRegressor(),
        "KNN": KNeighborsRegressor(),
        "rf": RandomForestRegressor(),
        "xg": XGBRegressor(),
        "ab": AdaBoostRegressor()
      }

      params={
        "lR": {
          "fit_intercept": [True, False],
          "positive": [True, False]
        },
        "DT": {
          "criterion": ["squared_error", "friedman_mse"],
          "splitter": ["best", "random"],
          "max_depth": [None, 5, 10, 20],
          "min_samples_split": [2, 5, 10],
        },
        "KNN": {
          "n_neighbors": [3, 5, 7, 9],
          "weights": ["uniform", "distance"],
          "algorithm": ["auto", "ball_tree", "kd_tree"]
        },
        "rf": {
          "n_estimators": [100, 200, 300],
          "max_depth": [None, 10, 20],
          "min_samples_split": [2, 5],
          "max_features": ["sqrt", "log2"]
        },
        "xg": {
          "n_estimators": [100, 200],
          "learning_rate": [0.01, 0.1, 0.2],
          "max_depth": [3, 5, 7],
          "subsample": [0.7, 0.8, 1.0],
        },
        "ab": {
          "n_estimators": [50, 100, 200],
          "learning_rate": [0.01, 0.1, 1.0],
          "loss": ["linear", "square", "exponential"]
        }
      }
      model_report=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)

      best_model_score=max(sorted(model_report.values()))
      best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      best_model=models[best_model_name]

      if best_model_score<0.6:
        raise customException("no best model found")
      logging.info("best found model on both training and testing dataset")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      predicted=best_model.predict(x_test)
      r2_scores=r2_score(y_test,predicted)
      return r2_scores
    except Exception as e:
      raise customException(e,sys)