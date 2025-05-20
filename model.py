import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, PolynomialFeatures, Binarizer, KBinsDiscretizer, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
import tensorflow as tf


class Data_Frame_Manager:
    numerical_column = ['tenure','MonthlyCharges','TotalCharges']
    target_name = 'Churn'
    categorical_column = [
        'gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract','PaperlessBilling'
    ]
    one_hot_column = ['PaymentMethod']

    preprocessor = ColumnTransformer(
            [('categorical', OrdinalEncoder(), categorical_column),
            ('one_hot', OneHotEncoder(handle_unknown='ignore'), one_hot_column),
            ('numeric', StandardScaler(), numerical_column)
            ],
            remainder='passthrough'
            )

    def __init__(self, csv):
        self.X = 0
        self.y = 0
        self.X_train = 0
        self.X_val = 0
        self.y_train = 0
        self.y_val = 0
        self.X_test = 0
        self.y_test = 0
        self.y_train_cat = 0
        self.y_val_cat = 0
        self.y_test_cat = 0
        self.model = 0
        self.history = 0
        self.create_pandas(csv)
        self.clean_data_frame()
        self.split_data_frame()
        self.process_pipeline()
        self.split_categories()
        self.build_model()
        self.fit_model()

        
    def create_pandas(self, csv):
        self.df = pd.read_csv(csv)
        print(self.df)


    def clean_data_frame(self):
        self.df['Churn']= self.df['Churn'].apply(lambda x : 1 if x=='Yes' else 0)
        self.df['TotalCharges']=pd.to_numeric(self.df['TotalCharges'])
        self.df = self.df.dropna()
    
    def split_data_frame(self):
        self.X = self.df.drop(columns=["customerID","Churn"])
        self.y = self.df["Churn"]
        self.X_train_0, self.X_test, self.y_train_0, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_0, self.y_train_0, test_size=0.2, random_state=42, stratify=self.y_train_0
        )

    def process_pipeline(self):
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_val  = self.preprocessor.transform(self.X_val)
        self.X_test  = self.preprocessor.transform(self.X_test)


    def split_categories(self):
        self.y_train_cat = tf.keras.utils.to_categorical(self.y_train, 2)
        self.y_val_cat = tf.keras.utils.to_categorical(self.y_val, 2)
        self.y_test_cat  = tf.keras.utils.to_categorical(self.y_test,  2)

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
            loss='categorical_crossentropy',
            metrics=['accuracy'] 
        )

    def fit_model(self):
        self.history = self.model.fit(
            self.X_train, self.y_train_cat,
            validation_data=(self.X_val, self.y_val_cat),
            epochs=60,
            batch_size=16,
            verbose=1
        )



if __name__ == "__main__":
    df = Data_Frame_Manager('churn.csv')
    print(df.history)