import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, PolynomialFeatures, Binarizer, KBinsDiscretizer, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, PolynomialFeatures, Binarizer, KBinsDiscretizer, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder, OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,make_scorer, accuracy_score, f1_score, recall_score, classification_report
import seaborn as sns
from sklearn.model_selection import GridSearchCV,KFold
import warnings


class Data_Frame_Manager:
    numerical_column = ['tenure','MonthlyCharges','TotalCharges']
    target_name = 'Churn'
    categorical_column = [
        'gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract','PaperlessBilling'
    ]
    one_hot_column = ['PaymentMethod']

    def __init__(self, csv):
        self.df = self.create_pandas(csv)
        self.pandas_cleaning()
        self.train_test_split()
        
    def create_pandas(self, csv):
        self.df = pd.read_csv(csv)


    def pandas_cleaning(self):
        self.df['Churn']= self.df['Churn'].apply(lambda x : 1 if x=='Yes' else 0)
        self.df['TotalCharges']=pd.to_numeric(self.df['TotalCharges'])
        self.df = self.df.dropna()
    
    def train_test_split(self):
        self.X = df.drop(columns=["customerID","Churn"])
        self.y = df["Churn"]
        self.X_train_0, self.X_test, self.y_train_0, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_0, self.y_train_0, test_size=0.2, random_state=42, stratify=self.y_train_0
        )




class Data_Frame_Cleaning:
    def __init__(self, df:object):
        self.df=df

    def pandas_cleaning(self):
        self['Churn']= self['Churn'].apply(lambda x : 1 if x=='Yes' else 0)
        self['TotalCharges']=pd.to_numeric(self['TotalCharges'])
        self = self.dropna()
        return self
    
    



df = Data_Frame_Creation.create_pandas('Churn.csv')

