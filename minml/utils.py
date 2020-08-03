import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)
warnings.filterwarnings("ignore")
import re
import time
import sklearn
import pandas as pd
import pickle
import os
import mlflow
from pycm import ConfusionMatrix
from pycm.pycm_param import PARAMS_LINK
from sklearn.base import ClassifierMixin
from sklearn.utils.testing import all_estimators
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from inspect import getmembers, isfunction
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gendata():
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    return X, y


def getdata():
    X, y = gendata()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    return x_train, x_test, y_train, y_test


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print("Run time :: {} seconds".format(round((te - ts), 1)))
        return result

    return timed


def vprint(statment, **kwargs):
    if kwargs.get("verbose", True):
        print(statment)


def getall_sklearn_classifiers(**kwargs):
    classifiers = [
        est for est in all_estimators() if issubclass(est[1], ClassifierMixin)
    ]
    vprint(
        "Total scikit learn packages fetched from scikit learn  : {}".format(
            len(classifiers)
        ),
        **kwargs
    )
    return classifiers


def algonaming(**kwargs):
    argstring = "_".join(
        [
            str(key) + "-" + str(value)
            for key, value in zip(kwargs.keys(), kwargs.values())
        ]
    )
    # if len(argstring):
    return argstring
    return "None"


def clf_measures():
    return [o for o in getmembers(sklearn.metrics._classification) if isfunction(o[1])]


def check_folder_exits(path=None):
    if path == None:
        path = os.path.join(cwd, "models")
    if not os.path.exists(path):
        os.makedirs(path)


def savepickle(filename, model):
    try:
        pickle.dump(model, open(filename, "wb"))
    except Exception as e:
        print("Failed to save file to path: {}".format(e))


def dfarrange(df):
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    df = df.reset_index(drop=True)
    return df


def transform_metrics(metric):
    flow_metrics = dict()
    flow_params = dict()
    for key, value in metric.items():
        key = re.sub("[^0-9a-zA-Z]+", "_", key.replace("%", "p"))
        if type(value) == float:
            flow_metrics[key] = value
        else:
            flow_params[key] = str(value)
    return flow_metrics, flow_params
