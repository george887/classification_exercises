import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
from sklearn.model_selection import train_test_split

def prep_iris(irisdf):
    '''
    This function acquires and prepares the iris data from a local csv, default.
    Passing cached=False acquires fresh data from Codeup db and writes to csv.
    Returns the iris df with dummy variables encoding species.
    '''
    # use my aquire function to read data into a df from a csv file
    irisdf = get_iris_data(cached)
    cols_to_drop = ['species_id','measurement_id']
    irisdf = irisdf.drop(columns=cols_to_drop)
    irisdf = irisdf.rename({'species_name':'species'}, axis = 1)
    dummy_df = pd.get_dummies(irisdf[['species']], dummy_na=False)
    irisdf = pd.concat([irisdf, dummy_df], axis = 1)
    return irisdf

def prep_titanic(titanic_df):
    '''
    This function reads titanic data into a df from a csv file.
    Returns prepped train, validate, and test dfs
    '''
    # use my acquire function to read data into a df from a csv file
    df = get_titanic_data(cached)
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked using dummy columns
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns='deck')
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # impute mean of age into null values in age column
    train, validate, test = impute_age(train, validate, test)
    
    return train, validate, test