import logging
import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
import numpy as np
import os
import pandas as pd


def read_data(path, ext):
    """Reads raw data and returns as dataframes.

    Args:
        path (str): path to raw data
        ext (str): extension correspoinding to train or test
    Returns:
        tuple of dataframes

    """
    dem = pd.read_csv(path + 'demographics' + ext)
    exp = pd.read_csv(path + 'exposure' + ext)
    trans = pd.read_csv(path + 'transfers' + ext, low_memory=False)

    return dem, exp, trans


def clean_text_columns(df, col_list=None):
    """Basic text cleaning.

    Args:
        df (dataframe): dataframe
        col_list (list): list of text columns to be cleaned
    Returns:
        cleaned dataframe

    """
    # use object columns if none provided
    if col_list is None:
        col_list = df.loc[:, df.dtypes == object].columns
        
    # get rid of non letter characters
    df[col_list] = df[col_list].apply(lambda x: x.str.replace('[^a-zA-Z ]', ''))    
    # convert to all uppercase
    df[col_list] = df[col_list].apply(lambda x: x.astype(str).str.upper())
    # strip whitespace
    df[col_list] = df[col_list].apply(lambda x: x.str.strip())
    
    return df


def autocorrect(myword):
    """Correct mispelled words using levanstine distance.

    Args:
        myword (str): word to check spelling for
    Returns:
        word with minimum distance

    """
    nltk.download('words')
    correct_spellings = words.words()
    distance_dict = {}
    for word in correct_spellings:
        ed = nltk.edit_distance(myword, word)
        distance_dict[word] = ed
        if ed == 0:
            break

    return(min(distance_dict, key=distance_dict.get))


def occupation_grouping(occupation, other=None):
    """Narrow down occupations into predefined groups.

    Args:
        occupation (str): occupation to be grouped
        other (list): list of occupations to be grouped into "other"
    Returns:
        occupation group

    """
    tech_keywords = ['ENGINEER', 'IT', 'SOFTWARE', 'TECHNOLOGY', 'DATA', 'DEVELOPER', 'PROGRAMMER', 'TECH', 'WEB', 'COMPUTER', 
                        'DBA', 'SCIENTIST', 'INFO']
    law_keywords = ['LAWYER', 'ATTORNEY', 'LEGAL']
    medicine_keywords = ['PHYSICIAN', 'DOCTOR', 'SURGEON', 'DENTIST', 'PHARMACIST', 'MD', 'NURSE', 'RN', 'NURSING', 'MEDICAL']
    self_keywords = ['SELF']
    sales_keywords = ['SALES']
    retired_keywords = ['RETIRED', 'RETIREE', 'RETIRE']
    student_keywords = ['STUDENT']
    executive_keywords = ['CEO', 'CTO', 'COO', 'CPO', 'CFO', 'CIO', 'DIRECTOR', 'FOUNDER', 'PRESIDENT', 'EXECUTIVE', 'OWNER', 'VP']
    unemployed_keywords = ['UNEMPLOYED', 'NONE']
    finance_keywords = ['FINANCE', 'ACCOUNTANT', 'WEALTH', 'BANKER', 'BANK', 'FINANCIAL', 'CPA', 'ACCOUNTING']
    manager_keywords = ['MANAGER', 'MANAGEMENT', 'SUPERVISOR']
    real_estate_keywords = ['REAL ESTATE', 'REALTOR', 'BROKER']
    business_keywords = ['BUSINESS', 'BUSINESSMAN']

    if other:
        if occupation in other:
            return 'OTHER'
        else:
            return occupation
    else:
        if any(word in occupation for word in tech_keywords):
            return 'TECH'
        elif any(word in occupation for word in law_keywords):
            return 'LAW'
        elif any(word in occupation for word in medicine_keywords):
            return 'MEDICINE'    
        elif any(word in occupation for word in self_keywords):
            return 'SELF'
        elif any(word in occupation for word in sales_keywords):
            return 'SALES'   
        elif any(word in occupation for word in retired_keywords):
            return 'RETIRED'   
        elif any(word in occupation for word in student_keywords):
            return 'STUDENT'  
        elif any(word in occupation for word in executive_keywords):
            return 'EXECUTIVE'  
        elif any(word in occupation for word in finance_keywords):
            return 'FINANCE'  
        elif any(word in occupation for word in real_estate_keywords):
            return 'REAL ESTATE' 
        elif any(word in occupation for word in manager_keywords):
            return 'MANAGER'  
        elif any(word in occupation for word in business_keywords):
            return 'BUSINESS'  
        elif any(word in occupation for word in unemployed_keywords):
            return 'UNEMPLOYED'  
        elif occupation == 'NAN':
            return 'MISSING'
        else:
            return occupation


def clean_demographics(dem):
    """Clean and process demographics data.

    Args:
        dem (dataframe): demographics data
    Returns:
        updated demographic dataframe

    """
    # clean text columns
    dem = clean_text_columns(dem, col_list=['STATE_CODE', 'COUNTRY_CODE', 'OCCUPATION'])
    # apply occupation grouping based on keywords
    dem['OCCUPATION_GROUP'] = dem['OCCUPATION'].apply(occupation_grouping)
    # Group occupations that show up less than 10 times as 'OTHER'
    dem['dummy'] = 1
    occs = dem.groupby('OCCUPATION_GROUP').agg({'dummy':sum}).reset_index()
    other_keywords = list(occs[occs['dummy'] < 10].OCCUPATION_GROUP.unique())
    dem['OCCUPATION_GROUP'] = dem['OCCUPATION_GROUP'].apply(occupation_grouping, other=other_keywords)
    # not applicable state code (non US)
    dem.loc[(dem['STATE_CODE'] == 'NAN') & (dem['COUNTRY_CODE'] != 'NAN') & (dem['COUNTRY_CODE'] != 'US'), 'STATE_CODE'] = 'NON US'

    return dem


def clean_exposures(exp):
    """Clean and process exposure data.

    Args:
        exp (dataframe): exposure data
    Returns:
        updated exposure dataframe

    """
    # create dummy variables for cluster_category
    exp = exp.merge(pd.get_dummies(exp[['cluster_category']]), left_index=True, right_index=True)
    exp['dummy'] = 1

    # define aggregations
    agg_dict = {
                 'dummy':'sum', 
                 'SENT_INDIRECT_EXPOSURE':'sum',
                 'SENT_DIRECT_EXPOSURE':'sum',
                 'RECEIVED_INDIRECT_EXPOSURE':'sum',
                 'RECEIVED_DIRECT_EXPOSURE':'sum',
                 'cluster_category':'nunique',
                 'cluster_name':'nunique'
                }
    clust_dict = {}
    for clust in exp.cluster_category.unique():
        clust_dict['cluster_category_' + clust] = 'sum'
    # add dummy variables to agg dict
    agg_dict.update(clust_dict) 

    # aggregate to account level
    exp_agg = exp.groupby('EXCHANGE_ACCOUNT_ID').agg(agg_dict).rename(
                columns={
                    'dummy':'exposure_count',
                    'SENT_INDIRECT_EXPOSURE':'SENT_INDIRECT_EXPOSURE_SUM',
                    'SENT_INDIRECT_EXPOSURE':'SENT_INDIRECT_EXPOSURE_SUM',
                    'SENT_DIRECT_EXPOSURE':'SENT_DIRECT_EXPOSURE_SUM',
                    'RECEIVED_INDIRECT_EXPOSURE':'RECEIVED_INDIRECT_EXPOSURE_SUM',
                    'RECEIVED_DIRECT_EXPOSURE':'RECEIVED_DIRECT_EXPOSURE_SUM',
                    'cluster_category':'cluster_category_count',
                    'cluster_name':'cluster_name_count'
                }).reset_index()
    
    # create averages
    for col in [col for col in exp_agg.columns if '_SUM' in col]:
        exp_agg[col.replace('_SUM', "_AVG")] = exp_agg[col] / exp_agg['exposure_count']
        
    return exp_agg


def clean_transfers(transfers):
    """Clean and process transfer data.

    Args:
        transfers (dataframe): transfer data
    Returns:
        updated transfer dataframe

    """
    # drop rows with all nans
    transfers.dropna(axis = 0, how = 'all', inplace = True)
    # convert to int
    transfers = transfers.astype({'ACCOUNT_ID':np.int64})
    # convert to datetime
    transfers['TX_TIME'] =  pd.to_datetime(transfers['TX_TIME'], errors='coerce')
    # aggregate to account level and create features broken out by category
    transfer_agg = pd.pivot_table(data=transfers, index=transfers.ACCOUNT_ID, columns=transfers.TYPE, aggfunc=('sum','nunique'))
    transfer_agg.columns = transfer_agg.columns.to_series().str.join('_')
    transfer_agg = transfer_agg.reset_index()

    return transfer_agg


def main(input_path='../../data/raw/', output_path='../../data/interim/', train=True):
    """Orchestration of data cleaning and feature engineering.

    Args:
        input_path (str): path to raw data
        output_path (str): path to store output data
        train (bool): boolean corresponding to True if train data, False if test data
    Returns:
        None. Saves transformed data to output path.

    """
    # define extension
    if train:
        ext = '_train.csv'
        desc = 'train'
    else:
        ext = '_test.csv'
        desc = 'test'

    # get data
    print('Reading ' + desc + ' data...')
    dem, exp, trans = read_data(input_path, ext)

    # process data
    print('Processing ' + desc + ' data...')
    dem_processed = clean_demographics(dem)
    exp_processed = clean_exposures(exp)
    trans_processed = clean_transfers(trans)

    # output processed data
    print('Writing ' + desc + ' data...')
    dem_processed.to_csv(output_path + 'demographics' + ext, index=False)
    exp_processed.to_csv(output_path + 'exposure' + ext, index=False)
    trans_processed.to_csv(output_path + 'transfer' + ext, index=False)


if __name__ == "__main__":

    dirname, filename = os.path.split(os.path.abspath(__file__))

    # Create logging file if DNE otherwise append to it
    logging.basicConfig(filename=os.path.join(dirname, "../../logs/build_features.log"), 
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

    # prep train data
    main()
    logging.info("Train data processed.")

    # prep test data
    main(train=False)
    logging.info("Test data processed.")


    print('Complete!')
    logging.info("Feature engineering complete.")






