import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


# instantiating pretty printer
pp = pprint.PrettyPrinter()
# prints accuracy and displays confusion table as heatmap


def accuracy_confusion_heatmap(y_test, y_pred, target):
    '''
    plots a heatmap of confusion matrix for a two class classification
    args:
        y_test: array of target variable of test dataframe
        y_pred: array of predictions
        target: string of prediction target
    return:
        nothing, only shows the plot
    '''

    ax = plt.subplot()
    conf_mat = confusion_matrix(y_test, y_pred)
    # helper df for heatmap
    df_cm = pd.DataFrame(conf_mat, range(2), range(2))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={
                "size": 20}, fmt='d')  # font size
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.xaxis.set_ticklabels([target, 'not ' + target])
    ax.yaxis.set_ticklabels([target, 'not ' + target])
    plt.yticks(rotation=90)
    plt.show()

# =============================================================================
# DATA EVALUATION  BASELINE
# =============================================================================


def plt_feat_importance(clf, features, n, correlation):
    '''
    plots the n most important features of a RandomForestClassifier

    args:
        clf (object): a fitted RandomForestClassifier object
        features (array): a list of the names of the used features
        n (int): number of features to plot
        correlation (pandas Series): list of correlation of features with target variable
    return:
        nothing, only plots a bar plot of the n most important features
    '''
    ax = plt.subplot()
    feat_importances = pd.Series(clf.feature_importances_, index=features)
    feat_importances.nlargest(n).plot(kind='barh', color='blue')
    plt.xlabel("Mean Decrease Impurity", fontsize=18)

    for i, v in enumerate(feat_importances.nlargest(n)):

        feature = feat_importances.nlargest(n).index[i]
        cur_corr = correlation[feature]
        is_positive = cur_corr > 0
        cur_corr = str(round(cur_corr, 2))
        if is_positive:
            cur_corr = cur_corr.replace('0.', ' .')

        cur_corr = cur_corr.replace('0.', '.')
        ax.text(max(feat_importances), i, cur_corr,
                color='blue', va='center')

    ax.text(max(feat_importances) * 0.75, n + 1, 'correlation',
            color='blue')
    plt.tight_layout()
    plt.show()
   
