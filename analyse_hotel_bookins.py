
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
# descriptive statistics?

# =============================================================================
# DATA UNDERSTANDING
# =============================================================================

import pprint

import matplotlib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from plotting_func import accuracy_confusion_heatmap, plt_feat_importance
from random_parameters import random_grid

# instantiating classes and setting plotting parameters
matplotlib.rcParams['figure.figsize'] = (10, 4)
pp = pprint.PrettyPrinter()
labelencoder = LabelEncoder()

# read data
df = pd.read_csv('../../data/hotel_bookings.csv')
df_original = df.copy()  # keep original df so I don't need to reload df

# get first overview of the dataframe, shape, dtypes, missing values etc.
df.head()
df.shape
df.columns
df.dtypes
df.describe()
na_cols = list(df.columns[df.isna().mean() > 0])
df.corr()['is_canceled'].sort_values()

# prints relative frequency of missing values for each column
for col in na_cols:
    print(col)
    print(df[col].isna().sum())
    print(df[col].isna().sum() / df.shape[0])
    print(df[col].dtype)


# =============================================================================
# DATA PREPARATION
# =============================================================================

# drop 'company columns because it has over 93% missing values
df.drop(columns=['company'], inplace=True)

# impute missing information about number of children with mode
df['children'] = df['children'].fillna(int(df.children.mode()), inplace=True)
# check if it worked
df['children'].isna().sum()

# impute missing values of agent with the mean
df['agent'].fillna(df['agent'].mean(), inplace=True)

# aggregate missing values of country to 'no_country'
df['country'].fillna('no_country', inplace=True)

# check how many unique categorial values are in each non-numerical column
cat_cols = df.dtypes[df.dtypes == 'object']
cat_cols.index[0]
cat_cols
for col in cat_cols.index:
    print(col)
    print(df[col].nunique())

df.reservation_status_date

# remove variables that have can bias dataset, or have no information
df.drop(columns=['reservation_status_date'], inplace=True)
df.drop(columns=['reservation_status'], inplace=True)

# convert arrival months into integers
keys_list = list(df['arrival_date_month'].unique())
value_list = [6, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
d = {k: v for k, v in zip(keys_list, value_list)}
df['arrival_date_month'] = df['arrival_date_month'].map(d)


# encoding country as label
df['country'] = labelencoder.fit_transform(df['country'])
df.country
for col in cat_cols.index:
    print(col)
    if col == 'reservation_status_date' or col == 'country':
        print('continue')
        continue
    try:
        df[col] = labelencoder.fit_transform(df[col])
    except:
        continue

df.dtypes[df.dtypes == 'object']


# =============================================================================
# DATA MODELLING BASELINE
# =============================================================================

# to Q1: decision tree and random forest to classify canceled or not =========

# separating data into dependent and independent variable
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

# splitting data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

# Decision Tree as baseline ==================================================
# 1. instantiating fitting DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)

# 2. predicting and evaluating model
y_preds = clf_dt.predict(X_test)

# =============================================================================
# DATA EVALUATION  BASELINE
# =============================================================================

pp.pprint(metrics.classification_report(y_true=y_test, y_pred=y_preds))
accuracy_confusion_heatmap(y_preds, y_test)


# =============================================================================
# DATA MODELLING RandomForestClassifier
# =============================================================================

# RandomForestClassifier for more sophisticated predictions ==================
# 1. instantiating and fitting model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 2. predicting if guest will cancel his reservation
y_preds = clf.predict(X_test)

rf_random = RandomizedSearchCV(estimator=clf,
                               param_distributions=random_grid,
                               n_iter=100, cv=3,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, y_train)
import pickle
pickle.dump(rf_random,open('rf_random.pkl', 'wb'))
# Question 2 ### ===========================================================
# to Q2: random forest for variable importance
df.corr()['is_canceled'].sort_values()
plt_feat_importance(clf, X.columns, 10)
important_features = pd.Series(clf.feature_importances_, index=X.columns)
relevant_features = list(important_features.sort_values(ascending=False)
                         .index[1:4])
relevant_features.append('is_canceled')
df[relevant_features].corr()['is_canceled']
df.groupby('country').mean()['is_canceled'].sort_values()

# =============================================================================
# DATA EVALUATION  BASELINE RandomForestClassifier
# =============================================================================

pp.pprint(metrics.classification_report(y_true=y_test, y_pred=y_preds))
accuracy_confusion_heatmap(y_preds, y_test)

# =============================================================================
# DATA MODELLING Question 3
# =============================================================================

# to Q3: Clustering?, Correlation? descriptive statistics?


# =============================================================================
# EVALUATION
# =============================================================================
