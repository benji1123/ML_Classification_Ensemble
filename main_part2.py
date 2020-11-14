# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt

import helper_part2 as helper

VERBOSE = False
SEED = 878
MAX_LEAVES = 400
ENSEMBLE_RANGE = (50, 1001, 19)


# data downloaded from https://archive.ics.uci.edu/ml/datasets/spambase
spam_data = np.loadtxt(fname="spambase.data", delimiter=",")
np.random.shuffle(spam_data)

# split into test/training sets
training_set_size = int(len(spam_data) * 0.7)
training_set = spam_data[:training_set_size]
test_set = spam_data[training_set_size:]
X_train = training_set[:, :-1]
y_train = training_set[:, -1]
X_test = test_set[:, :-1]
y_test  = test_set[:, -1]
data = (X_train, y_train, X_test, y_test)


## WHAT TO RUN
run = {
       "decision tree": True,
       "bagging": False,
       "random forest": False,
       "adaboost maxdepth1": False,
       "adaboost maxleaves10": False,
       "adaboost nolims": False,
       }

    
## DECISION TREE
if run["decision tree"]:
    print("analysing decision tree...")
    best_error, best_num_leaves, elapsed, xy, cv_score = helper.optimize_classifier(
        data,
        verbose=VERBOSE, 
        get_error=helper.init_and_eval_decision_tree, 
        indep_var_range=(2, 401),
        indep_var_name="max_num_leaves",
        get_crossval=helper.get_crossval_decision_tree,
    )
    print(f"DECISION TREE"
          f"best_test_err={best_error} " 
          f"with max_leaves={best_num_leaves}")
    plt.plot(xy[0], xy[1])
    plt.title("decision tree num_leaf_nodes vs test_error")
    plt.show()
    # cross val
    plt.plot(xy[0], cv_score)
    plt.title("decision tree num_leaf_nodes vs cross val")
    
    


## 50 BAGGING CLASSIFIERS
if run["bagging"]:
    print("analyzing bagging classifiers...")
    best_error, best_num_estimators, elapsed, xy, _ = helper.optimize_classifier(
        data,
        verbose=VERBOSE, 
        get_error=helper.init_and_eval_bagging_classifier, 
        indep_var_range=ENSEMBLE_RANGE,
        indep_var_name="num_estimators",
    )
    print(f"\n50 BAG CLASSIFIERS\n"
          f"total_elapsed={elapsed} sec\n"
          f"best test_err={best_error}\n"
          f"best num_estimators={best_num_estimators}")
    plt.plot(xy[0], xy[1])
    plt.title("bagging num_estimators vs test_error")
    plt.show()
    

## 50 RANDOM FORESTS
if run["random forest"]:
    print("analyzing random forests...")
    best_error, best_num_estimators, elapsed, xy, _ = helper.optimize_classifier(
        data,
        verbose=VERBOSE, 
        get_error=helper.init_and_eval_random_forest, 
        indep_var_range=ENSEMBLE_RANGE,
        indep_var_name="num_estimators",
    )
    print(f"\n50 RANDOM FORESTS\n"
          f"total_elapsed={elapsed} sec\n"
          f"best test_err={best_error}\n"
          f"best num_estimators={best_num_estimators}")
    plt.plot(xy[0], xy[1])
    plt.title("random forest num_estimators vs test_error")
    plt.show()

## 50 ADABOOST CLASSIFIERS (max-depth = 1)
if run["adaboost maxdepth1"]:
    print("analyzing adaboost maxdepth=1...")
    best_error, best_num_estimators, elapsed, xy, _ = helper.optimize_classifier(
        data,
        verbose=VERBOSE, 
        get_error=helper.init_and_eval_adaboost_maxleaves10, 
        indep_var_range=ENSEMBLE_RANGE,
        indep_var_name="num_estimators",
    )
    print(f"\n50 ADABOOST CLASSIFIERS (max-depth=1)\n"
          f"total_elapsed={elapsed} sec\n"
          f"best test_err={best_error}\n"
          f"best num_estimators={best_num_estimators}")
    plt.plot(xy[0], xy[1])
    plt.title("adaboost maxdepth1 num_estimators vs test_error")
    plt.show()
    

## 50 ADABOOST CLASSIFIERS (max-leaves = 10)
if run["adaboost maxleaves10"]:
    print("analyzing adaboost maxleaves=10...")
    best_error, best_num_estimators, elapsed, xy, _ = helper.optimize_classifier(
        data,
        verbose=VERBOSE, 
        get_error=helper.init_and_eval_adaboost_maxleaves10, 
        indep_var_range=ENSEMBLE_RANGE,
        indep_var_name="num_estimators",
    )
    print(f"\n50 ADABOOST CLASSIFIERS (no limitations)\n"
          f"total_elapsed={elapsed} sec\n"
          f"best test_err={best_error}\n"
          f"best num_estimators={best_num_estimators}")
    plt.plot(xy[0], xy[1])
    plt.title("adabooth maxleaves10 num_estimators vs test_error")
    plt.show()
    

## 50 ADABOOST CLASSIFIERS (no limitations)
if run["adaboost nolims"]:
    print("analyzing adaboost nolims...")
    best_error, best_num_estimators, elapsed, xy, _ = helper.optimize_classifier(
        data,
        verbose=VERBOSE, 
        get_error=helper.init_and_eval_adaboost_nolim, 
        indep_var_range=ENSEMBLE_RANGE,
        indep_var_name="num_estimators",
    )
    print(f"\n50 ADABOOST CLASSIFIERS (no limitations)\n"
          f"total_elapsed={elapsed} sec\n"
          f"best test_err={best_error}\n"
          f"best num_estimators={best_num_estimators}")
    plt.plot(xy[0], xy[1])
    plt.title("adaboost nolims num_estimators vs test_error")
    plt.show()
    
