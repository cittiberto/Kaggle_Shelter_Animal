################################################################################
# Description: Automatic tuning of XGBoost classifier via grid search
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        mer 15 giu 2016 20:45:41 CEST
################################################################################

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

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

def modelFit(alg, X, y, useTrainCV=True, cvFolds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgbParams = alg.get_xgb_params()
        xgbParams['num_class'] = 5
        xgTrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgbParams,
                          xgTrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cvFolds,
                          stratified=True,
                          metrics={'mlogloss'},
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                                                             xgb.callback.early_stop(3)])
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm
    alg.fit(X, y, eval_metric='mlogloss')

    # Predict
    dtrainPredictions = alg.predict(X)
    dtrainPredProb = alg.predict_proba(X)

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(y, dtrainPredictions)
    print "Log Loss Score (Train): %f" % metrics.log_loss(y, dtrainPredProb)
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


def xgbParametersTuning(trainFile):

    print('-----------------------------------------')
    print('|    Kaggle - Shelter Animal Outcome    |')
    print('-----------------------------------------')

    # 1) Read training set
    print('>> Read training set')
    train = pd.read_csv(trainFile)

    # 2) Extract target attribute and convert to numeric
    print('>> Preprocessing')
    y_train = train['OutcomeType'].values
    le_y = LabelEncoder()
    y_train = le_y.fit_transform(y_train)
    train.drop('OutcomeType', axis=1, inplace=True)

    # 3) Convert categorical attributes to dummy variables
#    train = pd.get_dummies(train)
#    train.drop('Status_Unknown', axis=1, inplace=True)

    # 4) Extract features and target from training set
    X_train = train.values

    # 5) Estimate number of classifiers by cross-validation
    if False:
        xgb1 = XGBClassifier(learning_rate =0.1,
                            n_estimators=1000,
                            max_depth=5,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            scale_pos_weight=1,
                            objective='multi:softmax',
                            seed=4245)

        modelFit(xgb1, X_train, y_train)

    # Best n_estimators = 225

    # 6) Tune max_depth and min_child_weight
    if True:
        param_test1 = {'max_depth':[9,10],
                       'min_child_weight':[1]}

        gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1,
                                                          n_estimators=240,
                                                          max_depth=5,
                                                          min_child_weight=1,
                                                          gamma=0,
                                                          subsample=0.8,
                                                          colsample_bytree=0.8,
                                                          objective= 'multi:softmax',
                                                          nthread=4,
                                                          scale_pos_weight=1,
                                                          seed=4245),
                                param_grid = param_test1,
                                scoring='log_loss',
                                n_jobs=4,
                                iid=False,
                                cv=3,
                                verbose=1)

        gsearch1.fit(X_train, y_train)
        print(gsearch1.grid_scores_)
        print(gsearch1.best_params_)
        print(gsearch1.best_score_)

    # Best max_depth = 11
    # Best min_child_weight = 0.5

    # 7) Tune gamma
    if False:
        param_test1 = {'gamma':[i / 10.0 for i in xrange(5)]}

        gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1,
                                                          n_estimators=225,
                                                          max_depth=11,
                                                          min_child_weight=0.5,
                                                          gamma=0,
                                                          subsample=0.8,
                                                          colsample_bytree=0.8,
                                                          objective= 'multi:softmax',
                                                          nthread=4,
                                                          scale_pos_weight=1,
                                                          seed=4245),
                                param_grid = param_test1,
                                scoring='log_loss',
                                n_jobs=4,
                                iid=False,
                                cv=3,
                                verbose=1)

        gsearch2.fit(X_train, y_train)
        print(gsearch2.grid_scores_)
        print(gsearch2.best_params_)
        print(gsearch2.best_score_)

    # Best gamma = 0.0

    # 8) Re-estimate number of classifiers by cross-validation
    if False:
        xgb1 = XGBClassifier(learning_rate =0.1,
                            n_estimators=1000,
                            max_depth=11,
                            min_child_weight=0.5,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            scale_pos_weight=1,
                            objective='multi:softmax',
                            seed=4245)

        modelFit(xgb1, X_train, y_train)

    # Best n_estimators = 73

    # 9) Tune subsample and colsample_bytree
    if False:
        param_test3 = {'subsample': [0.55, 0.6, 0.65],
                       'colsample_bytree': [0.65, 0.7, 0.75]}

        gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1,
                                                          n_estimators=73,
                                                          max_depth=11,
                                                          min_child_weight=0.5,
                                                          gamma=0,
                                                          subsample=0.8,
                                                          colsample_bytree=0.8,
                                                          objective= 'multi:softmax',
                                                          nthread=4,
                                                          scale_pos_weight=1,
                                                          seed=4245),
                                param_grid = param_test3,
                                scoring='log_loss',
                                n_jobs=4,
                                iid=False,
                                cv=3,
                                verbose=1)

        gsearch3.fit(X_train, y_train)
        print(gsearch3.grid_scores_)
        print(gsearch3.best_params_)
        print(gsearch3.best_score_)

    # Best subsample = 0.
    # Best colsample_bytree = 0.

    # 10) Tune regularization parameters
    if False:
        param_test4 = {'reg_alpha': [0, 0.00001, 0.01, 1]}

        gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1,
                                                          n_estimators=73,
                                                          max_depth=11,
                                                          min_child_weight=0.5,
                                                          gamma=0,
                                                          subsample=0.6,
                                                          colsample_bytree=0.7,
                                                          objective= 'multi:softmax',
                                                          nthread=4,
                                                          scale_pos_weight=1,
                                                          seed=4245),
                                param_grid = param_test4,
                                scoring='log_loss',
                                n_jobs=4,
                                iid=False,
                                cv=3,
                                verbose=1)

        gsearch4.fit(X_train, y_train)
        print(gsearch4.grid_scores_)
        print(gsearch4.best_params_)
        print(gsearch4.best_score_)

    # Best reg_alpha = 0

    # Re-fit model with new parameters
    if False:
        xgb1 = XGBClassifier(learning_rate =0.01,
                             n_estimators=2000,
                             max_depth=11,
                             min_child_weight=0.5,
                             gamma=0,
                             subsample=0.6,
                             colsample_bytree=0.7,
                             # reg_alpha=50,
                             scale_pos_weight=1,
                             objective='multi:softmax',
                             seed=4245)

        modelFit(xgb1, X_train, y_train)




if __name__ == "__main__":
    # File paths
    trainFile = '../Input/preprocessed_train.csv'

    # XGBoostClassifier parameter tuning
    xgbParametersTuning(trainFile)
