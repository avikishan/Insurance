from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.logger import logging
from Insurance.config import TARGET_COLUMN
from Insurance.exception import InsuranceException
from Insurance import utils
import sys,os
import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import ks_2samp

class DataValidation:

    def __init__(self,data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info("***************Data Validation***************")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise InsuranceException(e,sys)

    
    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        try:

            threshold=self.data_validation_config.missing_threshold
            null_report=df.isna().sum()/df.shape[0]
            drop_column_name=null_report[null_report>threshold].index
            self.validation_error[report_key_name]=list(drop_column_name)
            df.drop(list(drop_column_name),axis=1,inplace=True)

            if len(df.columns)==0:
                return None
            return df

        except Exception as e:
            raise InsuranceException(e,sys)
        

    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            #pass #base data
            base_columns=base_df.columns
            current_columns=current_df.columns

            missing_columns=[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available]")
                    missing_columns.append(base_column)
            
            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False
            return True
        except Exception as e:
            raise InsuranceException(e,sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report = dict()

            base_columns=base_df.columns
            current_columns=current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]

                same_distribution = ks_2samp(base_data,current_data)

                if same_distribution.pvalue > 0.05:
                    #Null Hypothesis accepted
                    drift_report[base_column] = {
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":True
                    }
                else:
                    drift_report[base_column] = {
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }
            self.validation_error[report_key_name]=drift_report
        except Exception as e:
            raise InsuranceException(e,sys)

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Data Validation Inititated")
            base_df= pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN},inplace=True)

            logging.info("Drop missing values columns")
            logging.info("Base df")
            base_df=self.drop_missing_values_columns(base_df,report_key_name="Missing_values_within_base_dataset")
            logging.info("Train df")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("test df")
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df=self.drop_missing_values_columns(train_df,report_key_name="Missing_values_within_train_dataset")
            test_df=self.drop_missing_values_columns(test_df,report_key_name="Missing_values_within_test_dataset")

            exclude_columns=[TARGET_COLUMN]
            logging.info("Convert Columns Float")
            logging.info("Base df")
            base_df=utils.convert_columns_float(base_df,exclude_columns=exclude_columns)
            logging.info("Train df")
            train_df=utils.convert_columns_float(train_df,exclude_columns=exclude_columns)
            logging.info("test df")
            test_df=utils.convert_columns_float(test_df,exclude_columns=exclude_columns)

            logging.info("is required columns exists")
            logging.info("train df")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df,current_df=train_df,report_key_name="Missing_columns_within_train_dataset")
            logging.info("test df")
            test_df_columns_status=self.is_required_columns_exists(base_df=base_df,current_df=test_df,report_key_name="Missing_values_wtihin_test_dataset")
            print("********************************")
            #print(train_df_columns_status)
            #print("********************************")
            #print(test_df_columns_status)
            logging.info("Data Drift")
            logging.info("Train df")
            if train_df_columns_status:
                self.data_drift(base_df=base_df,current_df=train_df,report_key_name="Data_drift_within_train_dataset")
            logging.info("test df")
            if test_df_columns_status:
                self.data_drift(base_df=base_df,current_df=test_df,report_key_name="Data_drift_within_test_dataset")
            
            #Write your report
            logging.info("Write yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,data=self.validation_error)
            
            data_validation_artifact=artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info("Data Validation Successfully Finished")
            return data_validation_artifact

        except Exception as e:
            raise InsuranceException(e,sys)

