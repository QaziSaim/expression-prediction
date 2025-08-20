import os
import warnings
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import dagshub

import logging


dagshub.init(repo_owner = 'QaziSaim',repo_name='AI-POWERED',mlflow=True)

logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual,predict):
    rmse = np.sqrt(mean_squared_error(actual,predict))
    mae = mean_absolute_error(actual,predict)
    r2 = r2_score(actual,predict)
    return rmse,mae,r2
 
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(40)
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url,sep=";")
    
    except Exception as e:
        logger.exception(
            'unable to download trainig & testing csv, check your internet connection. Error: %s',e
        )
    X = data.drop(columns='quality')
    y = data['quality']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha,l1_ratio = l1_ratio,random_state=42)
        lr.fit(X_train,y_train)
        predict_qualities = lr.predict(X_test)
        (rmse,mae,r2) = eval_metrics(y_test,predict_qualities)
        
        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        mlflow.log_param('alpha',alpha)
        mlflow.log_param('l1_ratio',l1_ratio)
        mlflow.log_param('rmse',rmse)
        mlflow.log_param('mae',mae)
        mlflow.log_param('r2',r2)
        
        # For remote server only (Dagshub)
        remote_server_uri = "https://dagshub.com/QaziSaim/AI-POWERED.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)



        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
           