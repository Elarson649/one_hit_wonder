from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from collections import defaultdict
import xgboost as xgb
import numpy as np
import imblearn.over_sampling
import keras


def score_model(y_train, y_train_predict, y_train_proba, y_val, y_val_predict, y_val_proba, score_dict):
    """
    Scores each round of cross-validation and appends the results to a list. Later, we take the mean of this list.
    :param y_train: the actual target of the training set
    :param y_train_predict: the predicted target of the training set
    :param y_train_proba: the prediction, from 0 to 1, of the training set
    :param y_val: the actual target of the validation set
    :param y_val_predict: the predicted target of the validation set
    :param y_val_proba: the prediction, from 0 to 1, of the validation set
    :param score_dict: a dictionary where the keys are metrics and the values are a list of values from each k-fold
    :return: the updated dictionary of results
    """
    score_dict['accuracy_scores_train'].append(accuracy_score(y_train, y_train_predict))
    score_dict['f1_scores_train'].append(f1_score(y_train, y_train_predict))
    score_dict['roc_auc_scores_train'].append(roc_auc_score(y_train, y_train_proba))
    score_dict['recall_scores_train'].append(recall_score(y_train, y_train_predict))
    score_dict['precision_scores_train'].append(precision_score(y_train, y_train_predict))
    score_dict['accuracy_scores_val'].append(accuracy_score(y_val, y_val_predict))
    score_dict['f1_scores_val'].append(f1_score(y_val, y_val_predict))
    score_dict['roc_auc_scores_val'].append(roc_auc_score(y_val, y_val_proba))
    score_dict['recall_scores_val'].append(recall_score(y_val, y_val_predict))
    score_dict['precision_scores_val'].append(precision_score(y_val, y_val_predict))
    return score_dict


def class_model(model, X_trainval, y_trainval, oversampling=0, scaling=0, pca_components=0, early_stopping_rounds=0,
                kfolds=5, threshold=0, epochs=100, callbacks=None, compiler=None):
    """
    Given a model and training/validation data for X and y, performs cross-validation and returns F-beta (beta is 2),
    recall, precision, accuracy, and ROC AUC.

    Args:
        model: An instance of an unfit model
        X_trainval: The features and observations to be used in training and validation.
        y_trainval: The corresponding target values for X_trainval
        oversampling: Specifies how much we want to oversample the positive target (the target represented by "1").
        Default is 0/oversampling is turned off.
        scaling: Specifies if we want to scale the data. Default is 0/scaling is turned off.

    Returns:
        model: The fitted instance of the model, fitted to the last crossfold
    """
    score_dict = defaultdict(lambda: [])
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=4)
    for train_ind, val_ind in kf.split(X_trainval, y_trainval):
        X_val = X_trainval.iloc[val_ind, :]
        y_val = y_trainval.iloc[val_ind]
        X_train = X_trainval.iloc[train_ind, :]
        y_train = y_trainval.iloc[train_ind]

        if scaling:
            std = StandardScaler()
            X_train = std.fit_transform(X_train)
            X_val = std.transform(X_val)

        if pca_components:
            pca = PCA(n_components=pca_components, random_state=4)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)

        if oversampling:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            ros = imblearn.over_sampling.RandomOverSampler(sampling_strategy={0: n_neg, 1: n_pos * oversampling},
                                                           random_state=4)
            X_train, y_train = ros.fit_sample(X_train, y_train)

        # XGB is separate because we need to set the model to the model found via early stopping
        if isinstance(model, xgb.XGBClassifier):
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, verbose=False, eval_set=eval_set,
                      eval_metric='logloss')
            y_train_proba = model.predict_proba(X_train, ntree_limit=model.best_ntree_limit)[:, 1]
            y_val_proba = model.predict_proba(X_val, ntree_limit=model.best_ntree_limit)[:, 1]
            y_train_predict = model.predict(X_train, ntree_limit=model.best_ntree_limit)
            y_val_predict = model.predict(X_val, ntree_limit=model.best_ntree_limit)

        # Need to compile the model each round of cross validation; otherwise, fit would use the parameters from the
        # last round
        elif isinstance(model, keras.engine.sequential.Sequential):
            model_copy = keras.models.clone_model(model)
            exec(compiler)
            y_train = np.array(y_train)
            model_copy.fit(X_train, y_train, epochs=epochs, callbacks=callbacks, verbose=0)
            y_train_proba = model_copy.predict(X_train)
            y_val_proba = model_copy.predict(X_val)
            y_train_predict = y_train_proba > .5
            y_val_predict = y_val_proba > .5

        # Linear SVC doesn't have predict_proba like other Sklearn algorithms
        elif isinstance(model, LinearSVC):
            model.fit(X_train, y_train)
            y_train_proba = model.decision_function(X_train)
            y_val_proba = model.decision_function(X_val)
            y_train_predict = model.predict(X_train)
            y_val_predict = model.predict(X_val)

        else:
            model.fit(X_train, y_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_train_predict = model.predict(X_train)
            y_val_predict = model.predict(X_val)

        if threshold:
            precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_val,y_val_proba)
            fbeta_curve = (2 * precision_curve[1:] * recall_curve[1:]) / (precision_curve[1:] + recall_curve[1:])
            fbeta_curve[np.isnan(fbeta_curve)] = 0
            max_fbeta = np.amax(fbeta_curve)
            index = np.where(fbeta_curve == max_fbeta)[0][0]
            threshold = threshold_curve[index + 1]
            score_dict['threshold_val'].append(threshold)
            score_dict['f1_threshold_val'].append(max_fbeta)

        score_dict = score_model(y_train, y_train_predict, y_train_proba, y_val, y_val_predict, y_val_proba, score_dict)

    # Print all the results in score_dict

    for key in score_dict:
        print(key, np.mean(score_dict[key]))
    return model
