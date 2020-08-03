# MinML
minMl - Sklearn classifications models auto training and logging progress 

  - Fits 30+ sklean algorithms
  - generates 25+ metrics (almost all classification metrics)
  - Logs status with MLFLow


### Packages

Used few open source projects:

* pycm - multi-class confusion matrix library. 
* seaborn - plotting library.
* mlflow -platform for the end-to-end machine learning lifecycle.


Uninstall enum if incase facing issues while installing mlflow.

```sh
$ pip uninstall -y enum34
```

### Run

Example:
```sh
$ x_train, x_test, y_train, y_test = getdata()
$ path = os.getcwd()
$ clf = classify(path=path, name="mltest", log=True, labels=["a", "b"])
$ report, fullreport = clf.run(x_train, y_train, x_test, y_test)
```

To view mlflow ui:
```sh
$ cd ../mlruns/path
$ mlfow ui
```
Open Ui in broweser using 
```sh
127.0.0.1:5000
```



### Todos

 - CV,param gridsearch,tree based models
 - Regression Framework
 - Text Classification models, Sentiment Analyis , QA systems

License
----

MIT


**Free Software, Hell Yeah!**
