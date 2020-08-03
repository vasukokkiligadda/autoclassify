from minml.utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)


class classify:
    def __init__(
        self, labels, name="ml", log=False, path=None, verbose=True, save=True
    ):
        self.log = log
        self.expname = name
        self.verbose = verbose
        self.save = save
        self.path = path
        self.estimators = getall_sklearn_classifiers()
        self.scores = clf_measures()
        self.report = pd.DataFrame()
        self.full_report = pd.DataFrame()
        self.labels = labels
        self.mpath = os.path.join(self.path, "models")
        self.ppath = os.path.join(self.path, "plots")
        self.cpath = os.path.join(self.path, "reports")
        self.rpath = os.path.join(self.path, "reports/summary")
        self.rpath_indv = os.path.join(self.path, "reports/indv")

    @timeit
    def training(self, algo, x, y, *algoname, **kwargs):
        try:
            print("=" * 100)
            print("-" * 40, *algoname, "-" * 40)
            model = algo(**kwargs)
            model.fit(x, y)
            modelname = str(*algoname) + algonaming(**kwargs)
            return modelname, model
        except Exception as e:
            return None, None

    def genreport(self, acts, pred, algo):
        model_metrics = dict()
        model_metrics["model"] = algo

        for metric in self.scores:
            name = metric[0]
            try:
                score = metric[1](acts, pred)
                if score.dtype != "float":
                    continue
            except TypeError:
                continue
            except Exception as e:
                # print(e)
                continue
            model_metrics[name] = round(score, 2)
        tn, fp, fn, tp = metrics.confusion_matrix(acts, pred).ravel()
        model_metrics["true negatives"] = tn
        model_metrics["false positives"] = fp
        model_metrics["false negatives"] = fn
        model_metrics["true positives"] = tp
        return model_metrics

    def savefile(self, modelname, model):
        print("Saving model to : {}".format(self.path))
        check_folder_exits(self.mpath)
        filepath = os.path.join(self.mpath, modelname)
        savepickle(filepath, model)
        print("Saved sucessfully")

    def cmfplot(self, y_test, y_pred_test, algoname):
        matrix = confusion_matrix(y_test, y_pred_test)
        # matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        size = len(self.labels)
        # Build the plot
        x = lambda size: size * 2 if size > 5 else 10
        y = lambda size: size * 2 if size > 5 else 6
        plt.figure(figsize=(x(size), y(size)))
        sns.set(font_scale=1.4)
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws={"size": 10},
            cmap=plt.cm.Greens,
            linewidths=0.2,
            fmt="g",
        )

        tick_marks = np.arange(len(self.labels))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, self.labels, rotation=25)
        plt.yticks(tick_marks2, self.labels, rotation=0)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("CM - {}".format(algoname))
        # plt.tight_layout()
        check_folder_exits(self.ppath)
        plt.savefig("{}/{}.png".format(self.ppath, algoname), bbox_inches="tight")
        print("plot saved")

    def mlflow_logging(self, algoname, metrics=None, param=None, files=None):
        mlflow.set_experiment(experiment_name=self.expname)
        with mlflow.start_run(run_name=algoname):
            if metrics != None:
                mlflow.log_metrics(metrics)
            if param != None:
                mlflow.log_params(param)
            if files != None:
                for file in files:
                    mlflow.log_artifact(file)
        print("data logged in mlflow")

    def savefiles(self, pmetrics, cmetrics, algoname):
        check_folder_exits(self.rpath)
        check_folder_exits(self.rpath_indv)
        pmetricspath = os.path.join(self.rpath, algoname)
        pmetrics.to_csv(pmetricspath + ".csv")
        cmetricspath = os.path.join(self.rpath_indv, algoname)
        cmetrics.to_csv(cmetricspath + ".csv")
        return pmetricspath + ".csv", cmetricspath + ".csv"

    def gen_cf_report(self, y_actu, y_pred, algoname):
        cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred)
        mapping = dict(zip(cm.classes, self.labels))
        cm.relabel(mapping)
        # overall metrics
        pmetrics = dict()
        for i, j in cm.overall_stat.items():
            pmetrics[PARAMS_LINK[i]] = j
        pmetrics = pd.DataFrame(list(pmetrics.items()), columns=["Metric", "Score"])
        # class metrics
        class_metrics = dict()
        for i, j in cm.class_stat.items():
            class_metrics[PARAMS_LINK[i]] = j
        cmetrics = pd.DataFrame(class_metrics).T
        pfile, cfile = self.savefiles(cmetrics, pmetrics, algoname)
        if self.log:
            metrics, params = transform_metrics(cm.overall_stat)
            cfimage = "{}/{}.png".format(self.ppath, algoname)
            self.mlflow_logging(
                algoname, metrics=metrics, param=params, files=[cfile, pfile, cfimage]
            )
        return pmetrics, cmetrics

    def evalmodel(self, x_test, y_test, model, algo):
        pred = model.predict(x_test)
        class_report = metrics.classification_report(y_test, pred)
        print(class_report)
        r = self.genreport(y_test, pred, algo[0])
        self.report = pd.concat(
            [self.report, pd.DataFrame.from_dict(r, orient="index").T], axis=0
        )
        self.cmfplot(y_test, pred, algo[0])
        pmetrics, cmetrics = self.gen_cf_report(y_test, pred, algo[0])
        pmetrics = pmetrics.set_index("Metric").T
        pmetrics["Model"] = algo[0]
        self.full_report = pd.concat([self.full_report, pmetrics], axis=0)

    def run(self, x_train, y_train, x_test, y_test, **kwargs):
        iter = 1
        for algo in self.estimators:
            try:
                modelname, model = self.training(
                    algo[1], x_train, y_train, algo[0], **kwargs
                )
                if model == None:
                    continue
                self.evalmodel(x_test, y_test, model, algo)
                if self.save == True:
                    self.savefile(modelname, model)
            except ValueError:
                continue
            except Exception as e:
                print(e)
                continue
            iter = iter + 1
        self.report.to_csv(os.path.join(self.cpath, "metrics.csv"))
        self.full_report.reset_index(drop=True, inplace=True)
        self.full_report = dfarrange(self.full_report)
        self.full_report.to_csv(os.path.join(self.cpath, "allmetrics.csv"))
        return self.report, self.full_report


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = getdata()
    path = os.getcwd()
    clf = classify(path=path, name="mltest", log=True, labels=["a", "b"])
    report, fullreport = clf.run(x_train, y_train, x_test, y_test)
