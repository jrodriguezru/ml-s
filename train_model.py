import math
import json
import logging
import yaml
import argparse
import pickle

# General Imports
import pandas as pd

# SKLearn Imports

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report


def load_config(config_path):
    """
    The function `load_config` reads and loads a YAML configuration file from the specified path.
    
    :param config_path: The `config_path` parameter in the `load_config` function is a string that
    represents the file path to the configuration file that you want to load and parse. This file should
    contain configuration settings in a format that can be read and processed by the `yaml` library's
    `safe_load` function
    :return: The function `load_config` is returning the configuration data loaded from the file
    specified by the `config_path`.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_data(path):
    """
    The function `read_data` reads a CSV file from a specified path, separates the features and labels,
    and returns them.
    
    :param path: The `path` parameter in the `read_data` function is a string that represents the file
    path to the CSV file that contains the data you want to read. 
    :return: The function `read_data` is returning two values: `df_X` which is the DataFrame with the
    column "y" dropped, and `df_label` which is a Series containing only the "y" column from the
    original DataFrame.
    """
    df = pd.read_csv(path)

    df_X = df.drop("y", axis=1)
    df_label = df["y"]
    return df_X, df_label

def save_model(clf, model_location):
    """
    The function `save_model` saves a machine learning model to a specified file location using pickle
    in Python.
    
    :param clf: The `clf` parameter in the `save_model` function is typically a machine learning model
    that has been trained and is ready to be saved to a file. This model could be a classifier,
    regressor, or any other type of model that has been fitted to the training data
    :param model_location: The `model_location` parameter in the `save_model` function is the file path
    where you want to save the trained model. It should be a string representing the location where the
    model will be stored.
    """
    with open(model_location, 'wb') as file:
        pickle.dump(clf, file)
    print('Model saved')

def load_model(model_location):
    """
    The `load_model` function reads a saved machine learning model from a specified location using
    pickle.
    
    :param model_location: The `model_location` parameter in the `load_model` function is a string that
    represents the file path to the location where the machine learning model is saved.
    :return: The function `load_model` returns the classifier (clf) that is loaded from the file located
    at `model_location`.
    """
    with open(model_location, 'r') as file:
        clf = pickle.load(file)
    return clf

def train_model(df_X, df_label, model_location, model_params):
    """
    The function `train_model` trains a classification model using logistic regression with
    preprocessing steps such as imputation, scaling, and one-hot encoding, and evaluates the model
    performance using metrics like classification report, confusion matrix, and AUC score.
    This function content comes from the notebook 'Model_Training.ipynb'.
    
    :param df_X: `df_X` is the input DataFrame containing the features used for training the model
    :param df_label: is a Series containing the corresponding labels of the DataFrame df_X
    :param model_location: The `model_location` parameter in the `train_model` function is the location
    where the trained model will be saved.
    :param model_params: The `model_params` parameter contains information about the parameters to be 
    used in the model training process. 
    In this case, it includes a key `max_iter` which is used as a parameter for the `LogisticRegression`
    classifier
    """
    numeric_features = ["x1", "x2", "x4", "x5"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["x3", "x6", "x7"]
    categorical_transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", LogisticRegression(max_iter=model_params['max_iter']))]
    )

    RANDOM_STATE=1337

    X_train, X_test, y_train, y_test = train_test_split(
        df_X,
        df_label,
        random_state=RANDOM_STATE
        )

    clf.fit(X_train, y_train)

    save_model(clf, model_location)

    # tprobs = clf.predict_proba(X_test)[:, 1]
    # print(classification_report(y_test, clf.predict(X_test)))
    # print('Confusion matrix:')
    # print(confusion_matrix(y_test, clf.predict(X_test)))
    # print(f'AUC: {roc_auc_score(y_test, tprobs)}')

if __name__ == "__main__":

    # Getting the configuration file path.
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (YAML).")
    args = parser.parse_args()

    # Loding the configuration file
    config = load_config(args.config)

    data_location = config['data_path']
    model_params = config['model_params']
    model_location = config['save_model_path']

    # Reading the data form data_location, training the model using specified params from 
    # model_params and saving the model in model_location
    df_X, df_label = read_data(data_location)
    train_model(df_X, df_label, model_location, model_params)