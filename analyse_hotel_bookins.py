
# =============================================================================
# BUISINESS UNDERSTANDING
# =============================================================================
# Question 1 How well can we predict if a customer is going to cancel
# done

# Question 2 What are the most important factors for a customer to cancel
#   ML and correlation
# done

# Question 3 Which type of customer book more in advance
# ML predict machine learning?
# Clustering?
# correlation?
# descriptive statistics?

# =============================================================================
# DATA UNDERSTANDING
# =============================================================================

import pickle
import pprint

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from plotting_func import accuracy_confusion_heatmap, plt_feat_importance
from random_parameters import random_grid

# instantiating classes and setting plotting parameters
matplotlib.rcParams['figure.figsize'] = (10, 4)
pp = pprint.PrettyPrinter()
labelencoder = LabelEncoder()

# prediction target column
target = 'is_canceled'

# read data
df = pd.read_csv('../../data/hotel_bookings.csv')
df_original = df.copy()  # keep original df so I don't need to reload df

# get first overview of the dataframe, shape, dtypes, missing values etc.
df.head()
df.shape
df.dtypes
df.describe()
na_cols = list(df.columns[df.isna().mean() > 0])
df.corr()[target].sort_values()

# prints relative frequency of missing values for each column
for col in na_cols:
    print(col)
    print(df[col].isna().sum())
    print(df[col].isna().sum() / df.shape[0])
    print(df[col].dtype)


# =============================================================================
# DATA PREPARATION
# =============================================================================

# drop 'company' column because it has over 93% missing values
# drop id column of travel agent, because not too useful for this analyse
df.drop(columns=['company', 'agent'], inplace=True)

# impute missing information about number of children with mode
df['children'] = df['children'].fillna(int(df.children.mode()), inplace=True)

# aggregate missing values of country to 'no_country'
df['country'].fillna('no_country', inplace=True)

# check how many unique categorial values are in each non-numerical column
cat_cols = df.dtypes[df.dtypes == 'object']
cat_cols.index[0]
for col in cat_cols.index:
    print(col)
    print(df[col].nunique())

df.reservation_status_date

# remove variables that have can bias dataset, or have no information
df.drop(columns=['reservation_status_date'], inplace=True)
df.drop(columns=['reservation_status'], inplace=True)

if(target == 'is_repeated_guest'):
    df.drop(columns=['previous_bookings_not_canceled'], inplace=True)
    df.drop(columns=['previous_cancellations'], inplace=True)

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

# check if transforming categorial columns worked
print(df.dtypes[df.dtypes == 'object'])

# check if all nas are removed from na_cols
for col in na_cols:
    if not col in df.columns:
        continue
    na_sum = df[col].isna().sum()
    print('The column {} has {} missing values'.format(col, na_sum))


# =============================================================================
# DATA MODELLING BASELINE
# =============================================================================

# to Q1: decision tree and random forest to classify canceled or not =========

# separating data into dependent and independent variable
X = df.drop(columns=[target])
y = df[target]

# splitting data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

# Decision Tree as baseline ==================================================
# 1. instantiating fitting DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)

# 2. predicting and evaluating model
y_pred_dt = clf_dt.predict(X_test)

# =============================================================================
# DATA MODELLING RandomForestClassifier
# =============================================================================

# RandomForestClassifier for more sophisticated predictions ==================
# 1. instantiating and fitting model
rf_not_tuned = RandomForestClassifier(random_state=42)
rf_not_tuned.fit(X_train, y_train)

# 2. predicting if guest will cancel his reservation
y_pred_non_tuned = rf_not_tuned.predict(X_test)


# Tune RandomForestClassifier with Random Grid Search
# rf_for_tuning = RandomForestClassifier(random_state=42)
# rf_random = RandomizedSearchCV(estimator=rf_for_tuning,
#                                param_distributions=random_grid,
#                                n_iter=100, cv=3,
#                                verbose=2,
#                                random_state=42,
#                                n_jobs=-1)
# # Fit the random search model
# rf_random.fit(X_train, y_train)
# best_random = rf_random.best_estimator_  # get best model
# # safe model
# pickle.dump(rf_random, open('rf_random.pkl', 'wb'))
# rf_random = pickle.load(open('./rf_random.pkl', 'rb'))

# # predict cancellation with best random forest
# y_pred_tuned = best_random.predict(X_test)
# =============================================================================

# DATA EVALUATION
# =============================================================================
# evaluation decision tree
pp.pprint(metrics.classification_report(y_true=y_test, y_pred=y_pred_dt))
accuracy_confusion_heatmap(y_pred_dt, y_test, target)

# non tuned Random Forest evaluation
pp.pprint(metrics.classification_report(
    y_true=y_test, y_pred=y_pred_non_tuned))
accuracy_confusion_heatmap(y_pred_non_tuned, y_test, target)


# Tuned Random Forest evaluation
# pp.pprint(metrics.classification_report(y_true=y_test, y_pred=y_pred_tuned))
# accuracy_confusion_heatmap(y_pred_tuned, y_test, target)

# Question 2 ### ===========================================================
# to Q2: random forest for variable importance

important_features = pd.Series(
    rf_not_tuned.feature_importances_, index=X.columns)
relevant_features = list(important_features.nlargest(10).index)
relevant_features.append(target)
correlation = df[relevant_features].corr()[target]
correlation.sort_values()
plt_feat_importance(rf_not_tuned, X.columns, 10, correlation)


# =============================================================================
# DATA MODELLING Question 3
# =============================================================================
df_original.groupby('distribution_channel').mean()['is_repeated_guest']
repeated_guest_channel_mean = df_original.groupby('distribution_channel')\
    .mean()['is_repeated_guest']
repeated_guest_channel_sum = df_original.groupby('distribution_channel')\
                                        .sum()['is_repeated_guest']
distribution_channel_df = pd.concat([repeated_guest_channel_mean,
                                     repeated_guest_channel_sum],
                                    axis=1)
distribution_channel_df.rename(columns={'is_repeated_guest':
                                        'Frequency_Repeated_Guest',
                                        'is_repeated_guest':
                                        'Sum_Repeated_Guest'},
                               inplace=True)
plt.table(cellText=distribution_channel_df)
distribution_channel_df
