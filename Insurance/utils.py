import pandas as pd
import numpy as np
import os
import sys
import yaml,dill
from Insurance.exception import InsuranceException
from Insurance.config import mongo_client
from Insurance.logger import logging

def get_collections_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    try:
        logging.info(f"Reading Data from database: {database_name} and collection: {collection_name}")
        df=pd.DataFrame(mongo_client[database_name][collection_name].find())
        logging.info(f"Found columnd: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping columns: _id")
            df=df.drop("_id",axis=1)
        logging.info(f"Rows and Columns in df: {df.shape}")
        return df
    except Exception as e:
        raise InsuranceException(e,sys)

def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise InsuranceException(e, sys)

def convert_columns_float(df:pd.DataFrame,exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtypes != 'O':
                    df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise InsuranceException(e,sys)
    
def save_object(file_path:str,obj:object)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_object:
            dill.dump(obj,file_object)
    except Exception as e:
        raise InsuranceException(e,sys)
    
def load_object(file_path:str,)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The {file_path} does not exists")
        with open(file_path,'rb') as file_object:
            return dill.open(file_object)
    except Exception as e:
        raise InsuranceException(e,sys)
    

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise InsuranceException(e,sys)