#Demo file for Batch prediction
#Training Pipeline

from Insurance.pipeline.batch_prediction import start_batch_prediction
from Insurance.pipeline.training_pipeline import start_training_pipeline

#file_path=r"D:\Machine_learning\End_to_End\Insurance\insurance.csv"

if __name__ == "__main__":
    try:
        
        #output=start_batch_prediction(input_file_path=file_path)
        output = start_training_pipeline()
        print(output)
    except Exception as e:
        print(e)