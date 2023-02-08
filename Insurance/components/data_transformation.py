from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
import sys,os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,LabelEncoder
#from sklearn.combine import SMOTE
import pandas as pd
import numpy as np
from Insurance.config import TARGET_COLUMN
from Insurance import utils
from Insurance.logger import logging

# Imputing the missing values
# Outliers handling
# Imbalanced Data Handling
# Convert categorical Data into Numerical Data

class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise InsuranceException(e,sys)
    
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant',fill_value=0)
            robust_scalar = RobustScaler()

            pipeline=Pipeline(steps=[
                ('Imputer',simple_imputer),
                ('Robust Scalar',robust_scalar)
            ])
            return pipeline
        except Exception as e:
            raise InsuranceException(e,sys)

    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:
        try:
            logging.info("******Data Transformation**********")
            logging.info("Reading Train File")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Reading test file")
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("input_features_train_df")
            input_features_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            logging.info("input_features_test_df")
            input_features_test_df = test_df.drop(TARGET_COLUMN,axis=1)

            logging.info("target_feature_train_df")
            target_feature_train_df=train_df[TARGET_COLUMN]
            logging.info("target_feature_test_df")
            target_feature_test_df=test_df[TARGET_COLUMN]

            logging.info("label_encoder")
            label_encoder=LabelEncoder()
            
            logging.info("target_feature_train_arr")
            target_feature_train_arr = target_feature_train_df.squeeze()
            logging.info("target_feature_test_arr")
            target_feature_test_arr = target_feature_test_df.squeeze()

            logging.info("Fit transform Train label Encoder")
            for col in input_features_train_df.columns:
                if input_features_train_df[col].dtype =='O':
                    input_features_train_df[col]=label_encoder.fit_transform(input_features_train_df[col])
                else:
                    input_features_train_df[col]=input_features_train_df[col]

            logging.info("Fit transform Test label Encoder")
            for col in input_features_train_df.columns:
                if input_features_test_df[col].dtype =='O':
                    input_features_test_df[col]=label_encoder.fit_transform(input_features_test_df[col])
                else:
                    input_features_test_df[col]=input_features_test_df[col]
            logging.info("data transformation pipeline")
            data_transformation_pipeline = DataTransformation.get_data_transformer_object()
            logging.info("data transformation pipeline fit")
            data_transformation_pipeline.fit(input_features_train_df)

            logging.info("input feattures_train_arr")
            input_features_train_arr=data_transformation_pipeline.transform(input_features_train_df)
            logging.info("input features test arr")
            input_features_test_arr= data_transformation_pipeline.transform(input_features_test_df)

            logging.info("np.c__")
            train_arr = np.c_[input_features_train_arr,target_feature_train_arr]
            test_arr = np.c_[input_features_test_arr,target_feature_test_arr]
            
            logging.info("save numpy array data")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_train_path,array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path,array=test_arr)

            logging.info("save object")
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,obj=data_transformation_pipeline)
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)
           
            logging.info("data transformation artifact")
            data_transformation_artifact=artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transform_train_path=self.data_transformation_config.transform_train_path,
                transform_test_path=self.data_transformation_config.transform_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path
            )
            logging.info("Data Validation succesfull")
            return data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e,sys)
