from Insurance.exception import InsuranceException
from Insurance.logger import logging
from Insurance.predictor import ModelResolver
from Insurance.utils import load_object
from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime
import os,sys



PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        model_resolver = ModelResolver(model_registry="saved_models")

        #Data Loading
        df=pd.read_csv(input_file_path)
        df.replace({"na":np.NAN},inplace=True)

        # Data Validation
        transorformer = load_object(file_path=model_resolver.get_latest_transform_path())
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

        #Encoding of the Datas
        input_features_name = list(transorformer.feature_names_in_)
        for i in input_features_name:
            if df[i].dtypes == 'object':
                df[i]=target_encoder.fit_transform(df[i])
        
        # Defining Input array
        input_array = transorformer.transform(df[input_features_name])
        ##print(model_resolver.get_latest_dir_path())
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_array)

        df['prediction']=prediction
        #print(os.path.basename(input_file_path))
        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_name = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_name,index=False,header=True)
        return prediction_file_name        
    except Exception as e:
        raise InsuranceException(e,sys)