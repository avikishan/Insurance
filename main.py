from Insurance.logger import logging
from Insurance.exception import InsuranceException
import os,sys
from Insurance.utils import get_collections_as_dataframe
from Insurance.entity.config_entity import DataIngestionConfig,DataValidationConfig
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evaluation import ModelEvaluation
# def test_logger_and_exception():
#     try:
#         logging.info("Startingt the test_logger_and_exception")
#         result=3/0
#         print(result)
#         logging.info("Ending point of the test_logger_and_exception")
#     except Exception as e:
#         logging.debug(str(e))
#         raise InsuranceException(e,sys)
    

if __name__=="__main__":
    try:
        #Data Ingestion
        # test_logger_and_exception()
        #get_collections_as_dataframe(database_name="INSURANCE",collection_name="INSURANCE_PROJECT")
        print("********************************************")
        print("Data Ingestion starting")
        training_pipeline_config=config_entity.TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config) 
        #print(data_ingestion_config.to_dict())
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
        print("Data Ingestion Done")
        print("***************************************")
        #Data Validation
        print("Data Validation Started")
        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        #print(data_ingestion_config.to_dict())
        data_validation=DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact)

        data_validation_artifact=data_validation.initiate_data_validation()
        print("Data Validation Done")
        print("******************************************")
        #print(data_validation_artifact)

        #Data Transformation
        print("Data Transformation started")
        data_transformation_config=config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation=DataTransformation(data_transformation_config=data_transformation_config,
                                               data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print("Data Transformation Done")

        print("***************************************")

        #Model Training
        print("Model Training")
        model_trainer_config=config_entity.ModelTrainingConfig(trainig_pipeline_config=training_pipeline_config)
        model_tainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_tainer.initiate_model_trainer()
        print("Model Taining Finished")
        print("************************************************")

        #Model Evaluation
        print("Model Evaluation Begins")
        model_eval_config=config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_eval=ModelEvaluation(model_evaluation_config=model_eval_config,
                                   data_ingestion_artifact=data_ingestion_artifact,
                                   data_transformation_artifact=data_transformation_artifact,
                                   model_trainer_artifact=model_trainer_artifact)
        model_eval_artifact=model_eval.initiate_model_evaluation()
        print("Model Evaluation Finished")
        print("***********************************************")
    except Exception as e:
        print(e)