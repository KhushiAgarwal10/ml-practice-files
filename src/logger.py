import logging
import os
from datetime import datetime

lOG_FILE=f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",lOG_FILE)
os.makedirs(logs_path,exist_ok=True)

lOG_FILE_PATH=os.path.join(logs_path,lOG_FILE)

logging.basicConfig(
  filename=lOG_FILE_PATH,
  format='[%(asctime)s]%(lineno)d %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)

