import lightgbm as lgb
import logging
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    dirname, filename = os.path.split(os.path.abspath(__file__))

    # Create logging file if DNE otherwise append to it
    logging.basicConfig(filename=os.path.join(dirname, "../../logs/train_model.log"), 
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')