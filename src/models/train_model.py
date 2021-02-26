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


def prepare_data(train=True, input_path='../../data/interim/', lgbm=True):
    """Prepare data for input into model by dropping and imputing data.

    Args:
        train (bool): boolean correspoing to train or predict
        input_path (str): path to cleaned and featurized data
        lgbm (bool): boolean correspoinding to model type
    Returns:
        Dataframe with relevant columns

    """
    if train:
        ext = '_train.csv'
        desc = 'train'
    else:
        ext = '_test.csv'
        desc = 'predict'

    # read interim data
    print('Loading ' + desc + ' data...')
    dem = pd.read_csv(input_path + 'demographics' + ext)
    exp = pd.read_csv(input_path + 'exposure' + ext)
    transfer = pd.read_csv(input_path + 'transfer' + ext)
    
    # combine dfs together
    df = dem.merge(exp, how='left', on='EXCHANGE_ACCOUNT_ID').merge(
            transfer, how='left', left_on='EXCHANGE_ACCOUNT_ID', right_on='ACCOUNT_ID')
    
    # define columns to drop
    drop_list = [x for x in df.columns if 'CURRENCY_sum' in x] # string concats
    drop_list.extend(['BIRTH_YEAR', 'EXCHANGE_ACCOUNT_ID', 'ACCOUNT_ID', 'OCCUPATION', 'dummy', 'CREATED_AT', 'FIRST_VERIFIED_AT'])
    cat_list = ['OCCUPATION_GROUP', 'STATE_CODE', 'COUNTRY_CODE'] 

    if lgbm:
        # convert to categorical
        df[cat_list] = df[cat_list].astype('category') 
    else:
        # drop categorical (other option is one-hot encode)
        drop_list.extend(cat_list)   
        # fill na with column mean      
        df = df.fillna(df.mean())
        
    df.drop(drop_list, axis=1, inplace=True)
    
    return df


def train_lgb_clf(X, y):
    """Train light gbm classifier.

    Args:
        X (dataframe): feature set
        y (series): dependent variable
    Returns:
        trained classifier

    """
    # format data for lgbm
    train_data = lgb.Dataset(X, label=y)

    # model parameters (needs further tuning)
    parameters = {
        'objective': 'multiclass',
        'num_class': 9,
        'metric': 'multi_logloss',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 30,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'boost_from_average': True,
        'verbose': -1,
        'seed': 100
    }
    
    # train model using specified parameters
    clf = lgb.train(parameters, train_data)

    return clf


if __name__ == "__main__":

    dirname, filename = os.path.split(os.path.abspath(__file__))

    # Create logging file if DNE otherwise append to it
    logging.basicConfig(filename=os.path.join(dirname, "../../logs/train_model.log"), 
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

    df = prepare_data()
    df.to_csv('../../data/processed/train.csv', index=False)
    logging.info("Done prepping and saving train data.")

    # Creating the dependent variable class
    df.level, definitions = pd.factorize(df['level'])
    # df.level = factor[0]
    # definitions = factor[1]

    # separate into dependent and independent
    y = df.pop('level')
    X = df
    columns = X.columns

    # train model
    print('Training model...')
    clf = train_lgb_clf(X, y)
    logging.info("Model training complete.")

    # save pickles
    print('Saving pickles...')
    pickle_path = os.path.join(dirname, '../../models')

    Pkl_Filename = "lgbm_model.pkl"  
    with open(os.path.join(pickle_path, Pkl_Filename), 'wb') as file:  
        pickle.dump(clf, file)
    logging.info("Model pickled and saved.")

    # Save trained columns as pickle for later use
    filename = 'train_columns.pk'
    with open(os.path.join(pickle_path, filename), 'wb') as file:
        pickle.dump(columns, file)
    file.close()
    logging.info("Save trained columns as pickle.")

    # Save level definitons
    filename = 'level_mappings.pk'
    with open(os.path.join(pickle_path, filename), 'wb') as file:
        pickle.dump(definitions, file)
    file.close()
    logging.info("Save level definitions.")

