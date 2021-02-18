# imports

from data import get_data
from data import clean_data
from encoders import DistanceTransformer
from encoders import TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe
        return pipe

    def run(self):
        """set and train the pipeline"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        
        self.pipeline.fit(X_train, y_train)
        # compute y_pred on the test set
        
        return X_train, X_test, y_train, y_test


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    print(df.shape)

    # clean data
    print("Clean data")
    df = clean_data(df)
    print(df.shape)
    
    # set X and y
    print("Set X and Y")

    y = df.pop("fare_amount")
    X = df
    t = Trainer(X, y)

    # hold out
    print("Holdout")
    
    pipe = t.set_pipeline()
    print(pipe)

    X_train, X_test, y_train, y_test = t.run()
    print(X_train.shape)
    # train

    # evaluate
    rmse = t.evaluate(X_test, y_test)
    print(rmse)
