from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
from Insurance.predictor import ModelResolver
import sys,os
import pandas as pd
import numpy as np
from Insurance.config import TARGET_COLUMN
from Insurance import utils
from Insurance.logger import logging

class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainingArtifact):
        try:
            self.model_evaluation_config=model_evaluation_config
            self.data_ingestion_config=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver()
        except Exception as e:
            raise InsuranceException(e,sys)
    
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("Model Evaluation Begins")
            latest_dir_path=self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_evaluation_artifact=artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=None
                )
                logging.info(f"Model Evaluation Finished from within {latest_dir_path}")
                logging.info(model_evaluation_artifact)
                return model_evaluation_artifact
            logging.info(f"Model Evaluation finished {latest_dir_path} exits")
    
        except Exception as e:
            raise InsuranceException(e,sys)