import os
import pickle
import sys
from src.exception import customException
from sklearn.metrics import r2_score
from src.exception import customException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
  try:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

    with open(file_path,'wb')as file_obj:
      pickle.dump(obj,file_obj)

  except Exception as e:
    raise customException(e,sys)
  

def evaluate_model(x_train,y_train,x_test,y_test,models,param):
  try:
    reprt={}
    for i in range(len(list(models))):
      model=list(models.values())[i]
      para=param[list(models.keys())[i]]
      
      gs=GridSearchCV(model,para,cv=3)
      gs.fit(x_train,y_train)

      model.set_params(**gs.best_params_)
      model.fit(x_train,y_train)
      # model.fit(x_train,y_train)
      y_train_pred=model.predict(x_train)
      y_test_pred=model.predict(x_test)
      train_model_score=r2_score(y_train,y_train_pred)
      test_model_score=r2_score(y_test,y_test_pred)

      reprt[list(models.keys())[i]]=test_model_score
    return reprt
  except Exception as e:
    raise customException(e,sys)
