"""Kaggle competition: Predicting a Biological Response.

Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

from __future__ import division
import numpy as np
import load_data
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import pandas as pd



def f():
    np.random.seed(0)  # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    X, y, X_submission = load_data.load()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    #skf = list(KFold(y, n_folds))
    skf = KFold(X.shape[0], n_folds, shuffle=True)
    
    clfs = [RandomForestRegressor(n_estimators=500, n_jobs=-1, criterion='mse'),
#            RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='mae'),
            ExtraTreesRegressor(n_estimators=500, n_jobs=-1, criterion='mse'),
#            ExtraTreesRegressor(n_estimators=100, n_jobs=-1, criterion='mae'),
            GradientBoostingRegressor(learning_rate=0.005, subsample=0.95, max_depth=4, n_estimators=500)]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            dataset_blend_train[test, j] = clf.predict(X_test)
            dataset_blend_test_j[:, i] = clf.predict(X_submission)
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print "Blending."
    #import pdb; pdb.set_trace()
    clf =  GradientBoostingRegressor(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)
    

    print "Saving Results."

    submission = pd.read_csv('data/sample_submission.csv')
    submission.iloc[:, 1] = clf.predict(dataset_blend_test)
    submission.to_csv('xgstacker_starter.sub.csv', index=None)
