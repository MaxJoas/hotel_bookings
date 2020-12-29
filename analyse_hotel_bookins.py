
# =============================================================================
# BUISINESS UNDERSTANDING
# =============================================================================
# Question 1 How well can we predict if a customer is going to cancel

# Question 2 What are the most important factors for a customer to cancel
#   ML and correlation

# Question 3 Which type of customer book more in advance
# ML predict machine learning?
# Clustering?
# correlation?
# descreptive statistics?

# =============================================================================
# DATA UNDERSTANDING
# =============================================================================

# this is mainly done already, still need to make some nice plots etc
import pandas as pd
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df = pd.read_csv('../../data/hotel_bookings.csv')
df.head()
df.shape
df.columns
df.dtypes
df.describe()
na_cols = list(df.columns[df.isna().mean() > 0])
for col in na_cols:
    print(col)
    print(df[col].isna().sum())
    print(df[col].isna().sum() / df.shape[0])
    print(df[col].dtype)

# =============================================================================
# DATA PREPARATION
# =============================================================================
del df['company']


# impute missing infomration about number of children with mode
df['children'] = df['children'].fillna(int(df.children.mode()), inplace=True)
df['children'].isna().sum()

# impute missing values of agent with the mean
df['agent'].fillna(df['agent'].mean(), inplace=True)

# aggreate missing values of country to 'no_countrie'
df['country'].fillna('no_country', inplace=True)

# check how many unique categorial values are in each non-numerical column
cat_cols = df.dtypes[df.dtypes == 'object']
cat_cols.index[0]
cat_cols
for col in cat_cols.index:
    print(col)
    print(df[col].nunique())

df.reservation_status_date


# convert arival months into integers
keys_list = list(df['arrival_date_month'].unique())
value_list = [6, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
d = {k: v for k, v in zip(keys_list, value_list)}
df['arrival_date_month'] = df['arrival_date_month'].map(d)

df['reservation_status_date'].dtype


# encoding country as lable
df['country'] = labelencoder.fit_transform(df['country'])
df.country
for col in cat_cols.index:
    print(col)
    if col == 'reservation_status_date' or col == 'country':
        print('continue')
        continue
    try:
        df = pd.concat([df.drop(columns=col), pd.get_dummies(df[col])], axis=1)
    except:
        df

df.shape
pd.get_dummies(df['customer_type'])
df.dtypes[df.dtypes == 'object']
# TODO reservation_status_date encoe? drop?
# 3 decide how to deal with categorila variables and handle them accordingly
# 4 potentially aggreagte the arrival date

# =============================================================================
# DATA MODELLING
# =============================================================================

# to Q1: decision tree and random forest to classifiy canceled or not
# to Q2: random forest for variable importance
# to Q3: Clusterint?, Correlation? descreptive statistics?

# =============================================================================
# EVALUATION
# =============================================================================
# MER
