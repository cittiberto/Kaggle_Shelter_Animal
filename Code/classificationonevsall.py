################################################################################
# Description: Kaggle competition - Shelter Animals Outcome
#              One-vs-ALl classification of shelter animal outcomes
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        ven 03 giu 2016 19:07:00 CEST
################################################################################

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

###################
# Import settings #
###################

from settings import *

##########################
# User-defined functions #
##########################

targetMapping = {'Adoption': 0,
                 'Died': 1,
                 'Euthanasia': 2,
                 'Return_to_owner': 3,
                 'Transfer': 4}

#######################
# Classification task #
#######################

def classification(trainFile, testFile, outputFile):

    print('-----------------------------------------')
    print('|    Kaggle - Shelter Animal Outcome    |')
    print('-----------------------------------------')

    # 1) Read training set
    print('>> Read training set')
    train = pd.read_csv(trainFile)

    # 2) Read test set
    print('>> Read test set')
    test = pd.read_csv(testFile, index_col=0)

    # 3) Extract target attribute and convert to numeric
    print('>> Preprocessing')
    y_train_dummies = pd.get_dummies(train['OutcomeType'])
    train.drop('OutcomeType', axis=1, inplace=True)

    # 4) Convert categorical attributes to dummy variables
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    train.drop('Status_Unknown', axis=1, inplace=True)
    test.drop('Status_Unknown', axis=1, inplace=True)

    # 4.1) Remove attributes that have been found to be not significant
    # notSignificant = ['ChocolateRappr',
                      # 'Gender_Unknown',
                      # 'OrangeRappr',
                      # 'RedRappr',
                      # 'Toy',
                      # 'NoColorRappr',
                      # 'Terrier',
                      # 'Hound',
                      # 'Non-Sporting',
                      # 'CreamRappr',
                      # 'Working',
                      # 'Sporting',
                      # 'Herding']

    # train.drop(notSignificant, axis=1, inplace=True)
    # test.drop(notSignificant, axis=1, inplace=True)
    print train.head()
    print y_train_dummies.head()
    print test.head()

    # 4) Extract features and target from training set
    X_train = train.values

    for outcomeType in y_train_dummies.columns:
        print('Classification of ' + outcomeType)
        # Initialize one-vs-all target
        y_train = y_train_dummies[outcomeType].astype(int)
        print(y_train)

        # 5) Split training set into reduced training set and validation set
        X_train_red, X_val, y_train_red, y_val = train_test_split(X_train,
                                                                  y_train,
                                                                  test_size=0.2,
                                                                  random_state=342,
                                                                  stratify=y_train)

        # 6) Training
        # 6.1) Initialize and train XGBoost xgbClassifier
        print('>> Train Xgboost classifier')

        xgbClassifier = xgb.XGBClassifier(max_depth=7,              # best: 7
                                          n_estimators=500,         # best: 500
                                          subsample=0.8,            # best: 0.8
                                          colsample_bytree=0.7,     # best: 0.7
                                          # objective='binary:logistic',
                                          n_class=2)

        xgbClassifier.fit(X_train_red,
                          y_train_red,
                          early_stopping_rounds=50,
                          eval_metric='logloss',
                          eval_set=[(X_val, y_val)])

        # 6.2) Initialize and train random forest
        rfClassifier = RandomForestClassifier(n_estimators=500,
                                            max_depth=15,
                                            verbose=0)

        rfClassifier.fit(X_train_red, y_train_red)

        # 6.3) Initialize and train neural networks
        # nnClassifier = MLPClassifier()
        # nnClassifier.fit(X_train_red, y_train_red)

        # 7) Validate
        # 7.1) Validate xgboost classifier
        print('>> Validate Xgboost classifier')
        y_val_proba = xgbClassifier.predict_proba(X_val)
        y_val_pred = xgbClassifier.predict(X_val)
        print('.. Classification report')
        print(classification_report(y_val, y_val_pred))
        print('.. Accuracy score')
        print(accuracy_score(y_val, y_val_pred))
        print('.. Log-loss score')
        print(log_loss(y_val, y_val_proba))

        # 7.2) Validate random forest classifier
        print('>> Validate random forest classifier')
        y_val_proba = rfClassifier.predict_proba(X_val)
        y_val_pred = rfClassifier.predict(X_val)
        print('.. Classification report')
        print(classification_report(y_val, y_val_pred))
        print('.. Accuracy score')
        print(accuracy_score(y_val, y_val_pred))
        print('.. Log-loss score')
        print(log_loss(y_val, y_val_proba))

        # 7.1) Validate neural network classifier
        # print('>> Validate neural network classifier')
        # y_val_proba = nnClassifier.predict_proba(X_val)
        # y_val_pred = nnClassifier.predict(X_val)
        # print('.. Classification report')
        # print(classification_report(y_val, y_val_pred))
        # print('.. Accuracy score')
        # print(accuracy_score(y_val, y_val_pred))
        # print('.. Log-loss score')
        # print(log_loss(y_val, y_val_proba))

        # 8) Variable importance
        importances = pd.DataFrame(xgbClassifier.feature_importances_,
                                index=train.columns,
                                columns=['Importance'])
        importances.sort_values('Importance', inplace=True, ascending=False)
        print (importances)
        # importances['Importance'].plot()
        # plt.show()

        importances = pd.DataFrame(rfClassifier.feature_importances_,
                                index=train.columns,
                                columns=['Importance'])
        importances.sort_values('Importance', inplace=True, ascending=False)
        print (importances)



if __name__ == "__main__":
    # File paths
    trainFile = '../Input/preprocessed_train.csv'
    testFile = '../Input/preprocessed_test.csv'
    outputFile = '../Output/'

    # Perform classification
    classification(trainFile, testFile, outputFile)
