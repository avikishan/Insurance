from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
from Insurance.predictor import ModelResolver
from sklearn.metrics import r2_score
import sys,os
import pandas as pd
import numpy as np
from Insurance.config import TARGET_COLUMN
from Insurance.logger import logging
from Insurance import utils

class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainingArtifact):
        try:
            self.model_evaluation_config=model_evaluation_config
            self.data_ingestion_artifact=data_ingestion_artifact
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
            
            
            # Find the previus model
            logging.info("Getting the file paths of transformer model and target encoder of previous model")
            transformer_path = self.model_resolver.get_latest_transform_path()
            model_path=self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            #previous model
            logging.info("loading previous transformer model and encoder")
            transformer=utils.load_object(file_path=transformer_path)
            model=utils.load_object(file_path=model_path)
            target_encoder=utils.load_object(file_path=target_encoder_path)

            #current model
            logging.info("loading current transformer model and encoder")
            current_transformer=utils.load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model=utils.load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder=utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            logging.info("Loading the test data")
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df=test_df[TARGET_COLUMN]
            y_true=target_df

            logging.info("Getting the input feature names")
            input_features_name = list(transformer.feature_names_in_)
            logging.info("Encoding the object type column")
            for i in input_features_name:
                if test_df[i].dtypes == 'object':
                    test_df[i]=target_encoder.fit_transform(test_df[i])
            logging.info("transforming the test df")
            input_arr=transformer.transform(test_df[input_features_name])
            logging.info("Predicting for test df")
            y_pred=model.predict(input_arr)
            logging.info("R2 score of previous model")
            #Comparisons nb/w new model and new model
            previous_model_score=r2_score(y_true=y_true,y_pred=y_pred)
            logging.info("Everything for current model")
            #Accuracy current model
            input_feature_name = list(current_transformer.feature_names_in_)
            curr_input_arr=current_transformer.transform(test_df[input_feature_name])
            curr_y_pred=current_model.predict(curr_input_arr)
            curr_y_true=target_df

            current_model_score=r2_score(y_true=curr_y_true,y_pred=curr_y_pred)
            logging.info("Final comparison")
            #Final comparison between both model
            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current model is not better than previous model")
            

            model_evaluation_artifact=artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=current_model_score-previous_model_score   
            )
                  
            logging.info(f"Model Evaluation finished {latest_dir_path} exits")
            return model_evaluation_artifact
        except Exception as e:
            raise InsuranceException(e,sys)
        

#cloud(AWS->s3buckets)
#Database->Model pusher