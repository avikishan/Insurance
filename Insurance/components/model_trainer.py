from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
import sys,os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from sklearn.combine import SMOTE
import pandas as pd
import numpy as np
from Insurance.config import TARGET_COLUMN
from Insurance import utils
from Insurance.logger import logging


#Model Define & Trainer
# 80% accuracy
# Accuracy in mew data:-60-80 set threshold accuracy>=70 to accept the new model
#Check for overfitting and underfitting

class ModelTrainer:
    def __init__(self,model_trainer_config:config_entity.ModelTrainingConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e,sys)
    
    def train_model(self,X,y):
        try:
            lr = LinearRegression()
            lr.fit(X,y)
            return lr
        except Exception as e:
            raise InsuranceException(e,sys)
        
    def initiate_model_trainer(self)->artifact_entity.ModelTrainingArtifact:
        try:
            logging.info("Model Training Begins")
            logging.info("Train data load")
            train_arr=utils.load_numpy_array(file_path=self.data_transformation_artifact.transform_train_path)
            logging.info("test data load")
            test_arr=utils.load_numpy_array(file_path=self.data_transformation_artifact.transform_test_path)

            logging.info("Spliting training data")
            x_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            logging.info("Spliting test data")
            x_test,y_test=test_arr[:,:-1],test_arr[:,-1]

            logging.info("model building")
            model=self.train_model(X=x_train, y=y_train)

            logging.info("Model predicttion for x_train")
            yhat_train=model.predict(x_train)
            r2_train_score = r2_score(y_true=y_train,y_pred=yhat_train)

            logging.info("Model prediction for x_test")
            yhat_test=model.predict(x_test)
            r2_test_score = r2_score(y_true=y_test,y_pred=yhat_test)

            logging.info("Condition checking mode expected accuracy and raising Exception")
            if r2_test_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"Model is not goot as it is not able give \
                                expected accuracy:{self.model_trainer_config.expected_accuracy}\
                                model actual score: {r2_test_score}")
            
            diff = abs(r2_train_score - r2_test_score)

            logging.info("Condition checking for overfitting and raising exception")
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train model and test score diff:{diff} is \
                                more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            logging.info("Save object for model")
            utils.save_object(file_path=self.model_trainer_config.model_path,obj=model)

            logging.info("Definingg Model Trainer Artifact")
            model_trainer_artifact=artifact_entity.ModelTrainingArtifact(
                model_path=self.model_trainer_config.model_path,
                r2_train_score=r2_train_score,
                r2_test_score=r2_test_score
            )

            logging.info("Model Training Finished")
            return model_trainer_artifact



        except Exception as e:
            raise InsuranceException(e,sys)