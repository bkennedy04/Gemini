import logging
import numpy as np
import os
import pandas as pd

def read_data(path='../../data/raw', train=True):
    """Reads raw data and returns as dataframes

    Args:
        path (str): path to raw data
        train (bool): boolean corresponding to True if train data, False if test data
    Returns:
        tuple of dataframes

    """
    if train:
    	ext = '_train.csv'
    else:
    	ext = '_test.csv'

    dem = pd.read_csv(path + 'demographics' + ext)
    exp = pd.read_csv(path + 'exposure' + ext)
    tran = pd.read_csv(path + 'transfers' + ext)


    return dem, exp, tran


if __name__ == "__main__":

    dirname, filename = os.path.split(os.path.abspath(__file__))

    # Create logging file if DNE otherwise append to it
    logging.basicConfig(filename=os.path.join(dirname, "../../logs/build_features.log"), level=logging.INFO)

