import os
import pickle
import sys
from src.exception import customException

def save_object(file_path,obj):
  try:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

    with open(file_path,'wb')as file_obj:
      pickle.dump(obj,file_obj)

  except Exception as e:
    raise customException(e,sys)