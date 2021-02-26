import lightgbm as lgb
import logging
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from train_model import prepare_data


def load_pickle(filename, path='../../models'):
    """Returns pickle file

    Args:
        filename (str): filename of pickle to load
        path (str): path to pickle file
    Returns:
        loaded pickle object

    """
    with open(os.path.join(pickle_path, filename), 'rb') as file:  
        my_pickle = pickle.load(file)
    file.close()

    return my_pickle


def get_missing_cols(train_columns, X):
    """Make sure test set has same columns as training set

    Args:
        train_columns (list): list of column names used in training
        X (dataframe): test dataframe
    Returns:
        Aligned test dataframe

    """
    # get missing columns in the training set
    missing_cols = set(train_columns) - set(X.columns)
    # add a missing column in test set as missing
    for c in missing_cols:
        X[c] = 0

    # filter to columns used in training
    X = X[train_columns]

    return X


if __name__ == "__main__":

    dirname, filename = os.path.split(os.path.abspath(__file__))

    # Create logging file if DNE otherwise append to it
    logging.basicConfig(filename=os.path.join(dirname, "../../logs/predict_model.log"), 
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

    pickle_path = os.path.join(dirname, '../../models')

    # prep test data
    print('Preping test data...')
    df = prepare_data(train=False)

    # load train columns
    print('Loading training columns...')
    train_columns = load_pickle('train_columns.pk')

    X = get_missing_cols(train_columns, df)
    X.to_csv('../../data/processed/test.csv', index=False)
    logging.info("Done prepping and saving test data.")

    print('Loading label definitions...')
    definitions = load_pickle('level_mappings.pk')

    # load pickled model
    print('Loading serialized model...')
    clf = load_pickle("lgbm_model.pkl")
    logging.info("Classifier loaded.")

    # predict
    print('Predicting...')
    y_pred = clf.predict(X)

    # get max probability for each prediction
    y_pred = [np.argmax(line) for line in y_pred]
    
    # convert back to labels
    pred_labels = definitions[y_pred]
    df['level'] = pred_labels

    logging.info("Predictions complete")

    df[['level']].to_csv('../../data/results/predictions.csv', index=False)
    print('Results saved.')
