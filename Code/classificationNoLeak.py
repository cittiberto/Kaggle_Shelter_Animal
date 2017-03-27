################################################################################
# Description: Kaggle competition - Shelter Animals Outcome
#              Classification of shelter animal outcome given various features
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
from sklearn.metrics import confusion_matrix, log_loss, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

###################
# Import settings #
###################

from settingsNoLeak import *

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
    y_train = train['OutcomeType'].map(targetMapping)
    train.drop('OutcomeType', axis=1, inplace=True)

    # 4) Extract features and target from training set
    X_train = train.values

    # 5) Split training set into reduced training set and validation set
    X_train_red, X_val, y_train_red, y_val = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=342,
                                                              stratify=y_train)

    # 6) Training
    # 6.1) Initialize and train XGBoost xgbClassifier
    print('>> Train Xgboost classifier')

    xgbClassifier = xgb.XGBClassifier(max_depth=7,              # best: 11
                                      min_child_weight=0.8,        # best: 0.5
                                      n_estimators=500,          # best: 500
                                      subsample=0.8,             # best: 0.8
                                      colsample_bytree=0.7,      # best: 0.7
                                      objective='multi:softprob')

    xgbClassifier.fit(X_train_red,
                      y_train_red,
                      early_stopping_rounds=50,
                      eval_metric='mlogloss',
                      eval_set=[(X_val, y_val)])

    # 6.2) Initialize and train random forest
    rfClassifier = RandomForestClassifier(n_estimators=1000,
                                          max_depth=10,
                                          verbose=1,
                                          n_jobs=4)

    rfClassifier.fit(X_train_red, y_train_red)

    # 6.3) Initialize and train neural networks
    # nnClassifier = MLPClassifier()
    # nnClassifier.fit(X_train_red, y_train_red)

    # 7) Validate
    # 7.1) Validate xgboost classifier
    print('>> Validate Xgboost classifier')
    y_val_proba = xgbClassifier.predict_proba(X_val)
    y_val_pred = xgbClassifier.predict(X_val)
    print('.. Confusion matrix')
    print(confusion_matrix(y_val, y_val_pred))
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
    print('.. Confusion matrix')
    print(confusion_matrix(y_val, y_val_pred))
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

    # 10) Fit the model to the entire training set
    print('>> Train classification model on entire training set')

    # Xgboost
    boosters=np.array([])
    xgb_params = {#'learning_rate': 0.2,
                  'max_depth': 8,
                  'n_estimators': 500,
                  'num_class': 5,
                  'objective': 'multi:softprob',
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'num_boost_rounds': 200,
                  'eval_metric': 'logloss',
                  'silent': '1'}

    #     dtrain = xgb.DMatrix(X_train_red, y_train_red)
    # dval = xgb.DMatrix(X_val, y_val)
    # watchlist = [(dtrain, 'train'), (dval, 'holdout')]

    for i in range(nXgboostClassifiers):
        print('.... Fitting XGB model #%d' % i)
        xgb_params['seed'] = np.random.randint(0,10000)

        # booster = xgb.train(xgb_params,
                            # num_boost_round=200,
                            # dtrain=dtrain,
                            # evals=watchlist,
                            # early_stopping_rounds=50,
                            # verbose_eval=True)
        X_train_red, X_val, y_train_red, y_val = train_test_split(X_train,
                                                                  y_train,
                                                                  test_size=0.1,  # best: 0.1
                                                                  random_state=np.random.randint(0,10000),
                                                                  stratify=y_train)

        booster = xgb.XGBClassifier(max_depth=7,              # best: 11
                                    min_child_weight=1,       # best: 1
                                    n_estimators=1000,        # best: 500
                                    subsample=0.8,            # best: 0.8
                                    colsample_bytree=0.7,     # best: 0.7
                                    seed=np.random.randint(0,10000),
                                    objective='multi:softprob')

        booster.fit(X_train_red,
                    y_train_red,
                    early_stopping_rounds=50,
                    eval_metric='mlogloss',
                    eval_set=[(X_val, y_val)])

        boosters = np.append(boosters, booster)


    # Random forests
    randomForests = np.array([])
    for i in range(nRandomForestClassifier):
        print('.... Fitting RF model #%d' % i)
        randomForest = RandomForestClassifier(n_estimators=500,
                                              criterion='gini',
                                              n_jobs=4)
        randomForest.fit(X_train, y_train)
        randomForests = np.append(randomForests, randomForest)

    # 11) Predict
    print('>> Predict outcomes of test set')
    X_test = test.values
    predictionsBoosters = []
    predictionsForests = []

    for c in boosters:
        # predictionsBoosters.append(c.predict(xgb.DMatrix(X_test)))
        predictionsBoosters.append(c.predict_proba(X_test))

    for c in randomForests:
        predictionsForests.append(c.predict_proba(X_test))

    # Average predictions
    y_test_proba_xgb = np.mean(predictionsBoosters, axis=0)
    y_test_proba_rf = np.mean(predictionsForests, axis=0)
    y_test_proba_bagging = np.mean(predictionsBoosters + predictionsForests, axis=0)

    # y_test_proba_nn = nnClassifier.predict_proba(X_test)

    # 13) Write predictions to submission file
    print('>> Write output')
    xgbPredictions = pd.DataFrame(y_test_proba_xgb,
                                  index=test.index,
                                  columns=['Adoption', 'Died', 'Euthanasia',
                                           'Return_to_owner', 'Transfer'])
    xgbPredictions.to_csv(outputFile + 'xgboostSubmissionNoLeak.csv')

    rfPredictions = pd.DataFrame(y_test_proba_rf,
                                 index=test.index,
                                 columns=['Adoption', 'Died', 'Euthanasia',
                                          'Return_to_owner', 'Transfer'])
    rfPredictions.to_csv(outputFile + 'rfSubmissionNoLeak.csv')

#     nnPredictions = pd.DataFrame(y_test_proba_nn,
                                 # index=test.index,
                                 # columns=['Adoption', 'Died', 'Euthanasia',
                                          # 'Return_to_owner', 'Transfer'])
#     nnPredictions.to_csv(outputFile + 'rfSubmission.csv')

    baggingPredictions = pd.DataFrame(y_test_proba_bagging,
                                      index=test.index,
                                      columns=['Adoption', 'Died', 'Euthanasia',
                                               'Return_to_owner', 'Transfer'])
    baggingPredictions.to_csv(outputFile + 'baggingSubmissionNoLeak.csv')


if __name__ == "__main__":
    # File paths
    trainFile = '../Input/preprocessed_train_noleak.csv'
    testFile = '../Input/preprocessed_test_noleak.csv'
    outputFile = '../Output/'

    # Perform classification
    classification(trainFile, testFile, outputFile)
