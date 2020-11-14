# -*- coding: utf-8 -*-
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score

SEED = 878


def optimize_classifier(
        data, verbose, get_error, indep_var_range, indep_var_name, get_crossval=None):
    """Test various versions of a classifier given a range of indep-vars."""
    best_error = 1
    best_indep_var = None
    start_time = time.time()
    plot_x = []
    plot_y = []
    crossval_scores = []
    for indep_var in range(*indep_var_range):
        error = get_error(*data, indep_var)
        if verbose:
            elapsed = int(time.time() - start_time)
            print(f"{indep_var_name}={indep_var} -> err={error}"
                  f" @ {elapsed} seconds")
        if error < best_error:
            best_error = error
            best_indep_var = indep_var
        plot_x.append(indep_var)
        plot_y.append(error)
        # cross val
        if get_crossval is not None:
            score = get_crossval(*data, indep_var)
            crossval_scores.append(score)
    # final results
    elapsed = int(time.time() - start_time)
    return best_error, best_indep_var, elapsed, (plot_x, plot_y), crossval_scores


def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    """Evaluate a classifier"""
    classifier.fit(X_train, y_train)
    test_error = 1 - classifier.score(X_test, y_test)
    return round(test_error, 6)


def init_and_eval_decision_tree(X_train, y_train, X_test, y_test, max_leaves):
    decision_tree = DecisionTreeClassifier(
        random_state=SEED, max_leaf_nodes=max_leaves)
    return evaluate_classifier(decision_tree, X_train, y_train, X_test, y_test)


def get_crossval_decision_tree(X_train, y_train, X_test, y_test, max_leaves):
    decision_tree = DecisionTreeClassifier(
        random_state=SEED, max_leaf_nodes=max_leaves)
    scores = cross_val_score(decision_tree, X_test, y_test, cv=5)
    return scores.mean()


def init_and_eval_bagging_classifier(
        X_train, y_train, X_test, y_test, num_estimators):
    bag = BaggingClassifier(
        n_estimators=num_estimators, 
        random_state=SEED)
    return evaluate_classifier(bag, X_train, y_train, X_test, y_test)


def init_and_eval_random_forest(
        X_train, y_train, X_test, y_test, num_estimators):
    """Train a random forest classifier and get its test error"""
    forest = RandomForestClassifier(
        n_estimators=num_estimators,
        random_state=SEED)
    return evaluate_classifier(forest, X_train, y_train, X_test, y_test)


def init_and_eval_adaboost_maxdepth1(
        X_train, y_train, X_test, y_test, num_estimators):
    """Train and evaluate AdaBoost classifier"""
    ab = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=num_estimators,
        random_state=SEED)
    return evaluate_classifier(ab, X_train, y_train, X_test, y_test)


def init_and_eval_adaboost_maxleaves10(
        X_train, y_train, X_test, y_test, num_estimators):
    """Train an AdaBoost classifier and get its test error"""
    ab = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_leaf_nodes=10),
        n_estimators=num_estimators,
        random_state=SEED)
    return evaluate_classifier(ab, X_train, y_train, X_test, y_test)


def init_and_eval_adaboost_nolim(
        X_train, y_train, X_test, y_test, num_estimators):
    """Train an AdaBoost classifier and get its test error"""
    ab = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=None),
        n_estimators=num_estimators,
        random_state=SEED)
    return evaluate_classifier(ab, X_train, y_train, X_test, y_test)
    