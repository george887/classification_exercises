import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
from sklearn.model_selection import train_test_split

def prep_iris(cached = True):
    '''
    This function acquires and prepares the iris data from a local csv, default.
    Passing cached=False acquires fresh data from Codeup db and writes to csv.
    Returns the iris df with dummy variables encoding species.
    '''
    # use my aquire function to read data into a df from a csv file
    df = get_iris_data(cached)
    cols_to_drop = ['species_id','measurement_id']
    df = df.drop(columns=cols_to_drop)
    df = df.rename({'species_name':'species'}, axis = 1)
    dummy_df = pd.get_dummies(df[['species']], dummy_na=False)
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def iris_split():

    train_validate, test = train_test_split(iris_df, test_size=.2, 
                                        random_state=123, 
                                        stratify=iris_df.species_name)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=iris_df.species_name)
   
    return train, validate, test



def prep_titanic(casched = True):
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