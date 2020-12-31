import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# instantiating pretty printer
pp = pprint.PrettyPrinter()
# prints accuracy and displays confusion table as heatmap


def accuracy_confusion_heatmap(y_test, y_pred):
    '''
    plots a heatmap of confusion matrix for a two class classificaitoion
    args:
        y_test: array of target variable of test dataframe
        y_pred: array of predictions
    return:
        nothing, only shows the plot
    '''

    conf_mat = confusion_matrix(y_test, y_pred)
    # helper df for heatmap
    df_cm = pd.DataFrame(conf_mat, range(2), range(2))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 20})  # font size
    plt.show()


def plt_feat_importance(clf, features, n):
    '''
    plots the n most important features of a RandomForestClassifier

    args:
        clf (object): a fitted RandomForestClassifier object
        features (array): a list of the names of the used features
        n (int): number of features to plot
    return:
        nothing, only plots a bar plot of the n most important features
    '''

    feat_importances = pd.Series(clf.feature_importances_, index=features)
    feat_importances.nlargest(n).plot(kind='barh')
    plt.xlabel("Mean Decrease Impurity", fontsize=18)
    plt.tight_layout()
    plt.show()
    return feat_importances
