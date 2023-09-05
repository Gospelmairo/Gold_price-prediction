import pandas as pd
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from typing import List, Tuple

def mae(y_true, y_pred):
    """
        @param y_true: (real) labels
        @param y_pred: predicted labels
    """
    return tf.metrics.mean_absolute_error(y_true=y_true, y_pred = y_pred)

def mse(y_true, y_pred):
    """
        @param y_true: (real) labels
        @param y_pred: predicted labels
    """
    return tf.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

# def plot_training_history(history:dict, plot_title:str, plot_xlabel:str="Epochs", plot_size:Tuple[int,int]=None,)->None:
#     # Looking at the content of the model training history
#     history_df = pd.DataFrame(history)
#     print("history_df.head()")
#     print(history_df.head())

#     # Model 9 performance
#     history_df.plot(figsize=plot_size) #(figsize=(8,6))

#     plt.xlabel(plot_xlabel)
#     plt.title(plot_title);


def plot_training_history(history:tf.keras.callbacks.History, plot_title:str, plot_xlabel:str="Epochs", plot_size:Tuple[int,int]=None,
                          metrics_to_plot:List[str]=None)->None:
    # Looking at the content of the model training history
    history_df = pd.DataFrame(history.history)
    print("history_df.head()")
    print(history_df.head())

    if metrics_to_plot is not None:
        history_df = history_df.loc[:, metrics_to_plot]

    # Model 9 performance
    history_df.plot(figsize=plot_size) #(figsize=(8,6))

    plt.xlabel(plot_xlabel)
    plt.title(plot_title);


def plot_predictions(train_X, train_labels,
                    test_X, test_labels,
                    predictions):
    """
        Plots training data, test data, and compares predictions to ground truth labels.
    """
    plt.figure( figsize=(10,7) )

    if train_X is not None and train_labels is not None:
        # Plot training data in blue 
        plt.scatter(train_X, train_labels, c="b", label="Training data")

    if test_X is not None and test_labels is not None:
        # Plot test data in green
        plt.scatter(test_X, test_labels, c="g", label="Test data")
    
    if test_X is not None and predictions is not None:
        # Plot prediction data
        plt.scatter(test_X,predictions,c="r", label="Predictions")

    #Show a legend
    plt.legend(); 

### ====================== ML functions ============================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

# Saving ML model https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-using-pickle

def search_best_regresion_models_by_cross_validation(algorithms:dict, X_train, y_train,
                                                     X_test, y_test, search_on_top_k:int=None)->pd.DataFrame :
    """
        return:
            - dataframe containing the result of each model
            - a list consisting of the best model for each algorithm
    """
    training_report = {
        "algo":[],
        "best_params_":[],
        "best_score_":[],
        "best_estimator_":[],
          
        "mae":[],
        "mse":[],
        "rmse":[],
        "r2_score":[],
        
#         #refer to my method train_model_by_crossvalidation()
#         "accuracies":[],
#         "accuracy":[],
#         "error":[],
        
#         "Name":[],
#         "Training AUC":[],
#         "Testing AUC":[],
#         "Recall":[],
#         "Precision":[],
#         "F1 Score":[],
#         "MSE":[]
    }
#     best_models=[]
    
    if search_on_top_k is not None:
        X_train = X_train[:search_on_top_k,:]
        y_train = y_train[:search_on_top_k] 
        X_test = X_test[:search_on_top_k,:]
        y_test = y_test[:search_on_top_k]
    
    for key, value in algorithms.items():
#         print("key")
#         print(key)
#         print("value")
#         print(value)
        algo = key
        print("========================================================================")
        print(f"============== Working on {algo} algorithm ==============")
        print("========================================================================")

        model = value.get("model")
        parameters = value.get("parameters")
        fold = value.get("fold")
        scoring = value.get("scoring")
        
#         print("Started best estimator research with GridSearchCV...")
        
        # Define search
        print("Defining GridSearchCV...")
        search = GridSearchCV(model,parameters, cv=fold, scoring=scoring, return_train_score=True )
        # Execute search
        print("Starting best estimator research with GridSearchCV...")
        search.fit(X_train, y_train) #result = search.fit(X_train, y_train)
        
        print("Found best estimator")
        
        training_report["algo"].append(algo)
        training_report["best_params_"].append(search.best_params_) 
        training_report["best_score_"].append(search.best_score_) 
        training_report["best_estimator_"].append(search.best_estimator_)
        
        print("Starting evaluating best estimator...")
        best_model = search.best_estimator_
        #best_model.fit(X_train, y_train)
        y_preds = best_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_preds)
        mse = mean_squared_error(y_test, y_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_preds)
        
        print("Evaluated best estimator.")
        
        training_report["mae"].append(mae)
        training_report["mse"].append(mse)
        training_report["rmse"].append(rmse)
        training_report["r2_score"].append(r2)
        print("Generated report.")
        print("\n \n")
    
    
    report_df = pd.DataFrame(training_report)  
#     print(report_df)
#    report_df
        
    return report_df #(report_df, best_models)


def train_regression_model_by_crossvalidation(model, X_train, y_train,
                                              X_test, y_test, kfold=None,
                                              n_folds:int=5, train_on_top_k:int=None, 
                                              verbose:bool=True)->Tuple:
    print("Training started")
    if kfold is None:
        kfold = KFold(n_splits=n_folds, shuffle = True, random_state=42)
    
    if train_on_top_k is not None:
        X_train = X_train[:train_on_top_k,:]
        y_train = y_train[:train_on_top_k] 
        X_test = X_test[:train_on_top_k,:]
        y_test = y_test[:train_on_top_k]
    
    scores = cross_val_score(model,X_train, y_train,cv=kfold, scoring="r2") # scoring="neg_mean_absolute_error"
    score = np.mean(scores)
    error = scores.std()
    
    if verbose:
        print("Training folds scores : {}".format(scores))
        print("Trained model score : {}".format(score))
        print("Trained model error : {}".format(error))
#         print("***"*2)
#         print("")    
    
    print("Training ended")
    print("************************"*3)
    print("\n")
    
    test_regression_model_by_crossvalidation(model=model,kfold=kfold,
                                             X_test=X_test, y_test=y_test)
    
    return (model, scores, score, error, kfold)


def test_regression_model_by_crossvalidation(model, kfold,
                                             X_test, y_test,
                                              verbose:bool=True)->Tuple[float,float,float,float]:
    
    y_preds = cross_val_predict(model,X_test,y_test,cv=kfold)
    #score = r2_score(y_preds,y_test)
    
    mae = mean_absolute_error(y_test, y_preds)
    mse = mean_squared_error(y_test, y_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_preds)
    
    if verbose:
        print("Trained model test MAE : {}".format(mae))
        print("Trained model test MSE : {}".format(mse))
        print("Trained model test RMSE : {}".format(rmse))
        print("Trained model test R2 SCORE : {}".format(r2))
        
        
    return (mae,mse,rmse,r2)











### ==================  ML functions (old) =========================
from sklearn.preprocessing import  OneHotEncoder ,LabelEncoder, OrdinalEncoder , MinMaxScaler, Normalizer, StandardScaler
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, mean_squared_error, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

def plot_confusion_matrix(confusion_matrix:np.ndarray, display_labels:list,
                         figsize_x:int=10, figsize_y:int=8):
    """
        Based on https://www.w3schools.com/python/python_ml_confusion_matrix.asp
        
        https://stackoverflow.com/questions/66483409/adjust-size-of-confusionmatrixdisplay-scikitlearn
    """
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix) #, display_labels=display_labels)
    
    #plt.figure(figsize=(figsize_x,figsize_y))
    
    fig, ax = plt.subplots(figsize=(figsize_x,figsize_y))
    
    cm_display.plot(ax=ax)
    plt.show()
    

def train_model(model, model_name:str,_x_train:np.ndarray, _x_test:np.ndarray, _y_train:np.ndarray, _y_test:np.ndarray, _training_report:dict, verbose=True)->dict:
    """
        _model is the instance of the model that must be trained (example : LogisticRegression(),
        while _model_name is its name (example : "LR")
    """
    
    print("Working on {} model".format(model_name))
    
    #training the model
    model.fit(_x_train, _y_train)
    
    predicted_y = model.predict(_x_train)
    train_auc = accuracy_score(predicted_y, _y_train) * 100

    predicted_y = model.predict(_x_test)
    test_auc = accuracy_score(predicted_y,_y_test) * 100
    recall = recall_score(predicted_y, _y_test, average="micro")
    precision = precision_score(predicted_y, _y_test, average="micro")
    f1 = f1_score(predicted_y, _y_test, average="micro")
    mse = mean_squared_error(predicted_y, _y_test)
    
    _training_report["Name"].append(model_name)
    _training_report["Training AUC"].append(train_auc)
    _training_report["Testing AUC"].append(test_auc)
    _training_report["Recall"].append(recall)
    _training_report["Precision"].append(precision)
    _training_report["F1 Score"].append(f1)
    _training_report["MSE"].append(mse)
    
    if verbose:
        print("Accurary on training set : {} %".format(train_auc))
        #TODO : print others testing result
        
    print("***********"*3)
    print("")
     
    return _training_report


def build_model_pipeline(_model, _model_name:str, n_features:int=6, verbose=True)->Pipeline:
    """
        _model is the instance of the model that must be trained (example : LogisticRegression(),
        while _model_name is its name (example : "LR")
    """
                         
    if verbose:
        print("Model building started for {} features...".format(n_features))
    
    #optimal features selection based on features union
    features = []
    features.append( ("PCA", PCA(n_components=n_features)) )
    features.append( ("SelectKBest", SelectKBest(k=n_features+1)) )
    feature_union = FeatureUnion(features)

    estimators = []
    estimators.append(("Features Union", feature_union))

    #preprocessing tasks
    estimators.append(("rescale", MinMaxScaler(feature_range=(0,1))))
    estimators.append(("normalize", Normalizer()))
    estimators.append(("standardize", StandardScaler()))
    
    estimators.append( (_model_name, _model ) ) 
    
    if verbose:
        print("Model building ended. \n")
        #print("")
    
    return Pipeline(estimators)

def train_model_by_crossvalidation(_model:Pipeline,_X:np.ndarray, _Y:np.ndarray,num_folds:int=5, verbose:bool=True)->Tuple:
    print("Training started")
    kfold = KFold(n_splits=num_folds, shuffle = True, random_state=10)
    #model = LogisticRegression()
    accuracies = cross_val_score(_model,_X,_Y,cv=kfold)
    accuracy = np.mean(accuracies)
    error = accuracies.std()
    if verbose:
        print("Training folds accuracies : {}".format(accuracies))
        print("Trained model accuracy : {}".format(accuracy))
        print("Trained model error : {}".format(error))
        print("***"*2)
        print("")
    
    print("Training ended")
    print("************************"*3)
    print("")
    
    return (_model, accuracies, accuracy, error, kfold)

def test_model_by_crossvalidation(_model:Pipeline, _X:np.ndarray, _Y:np.ndarray,_kfold, cm_display_labels:list,
                                  verbose_accuracy:bool=True, verbose_confusion_matrix:bool=True)->Tuple:
    Y_hat = cross_val_predict(_model,_X,_Y,cv=_kfold)
    accuracy = accuracy_score(Y_hat,_Y)
    conf_mat = confusion_matrix(_Y, Y_hat)
    if verbose_accuracy:
        print("Trained model test accuracy : {}".format(accuracy))
    if verbose_confusion_matrix:
        print("Confusion matrix : ")
        print(conf_mat)
        #plot_confusion_matrix(confusion_matrix=conf_mat,display_labels=[False,True])
        plot_confusion_matrix(confusion_matrix=conf_mat,display_labels=cm_display_labels)
    return conf_mat,accuracy



