import pandas as pd
import numpy as np
from sklearn.svm import SVC
from pathlib import Path
from pandas import DataFrame
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def solution():
    # Loading the data from NumberRecognitionBigger.mat file
    data = loadmat('NumberRecognitionBigger.mat')
    X_data = data['X']  # (28,28,30000)
    Y_label = data['y']  # (1, 30000)
    # Transposing X_data, Y_label
    X_data = X_data.transpose(2, 0, 1)  # (30000, 28, 28)
    Y_label = Y_label.transpose()  # (30000, 1)
    #  Reshaping X_data to (30000,784)
    X_data = X_data.reshape(X_data.shape[0], np.prod(X_data.shape[1:]))
    # Creating Data frame for X_data, Y_label
    X_data = pd.DataFrame(X_data)
    Y_label = pd.DataFrame(Y_label)
    df_XY = pd.concat([X_data, Y_label], axis=1)
    # Renaming last column with Label to avoid ambiguity
    df_XY.columns = [*df_XY.columns[:-1],  'Label']
    for val in range(8):
        df_XY = df_XY.drop(df_XY[df_XY.Label == val].index)
    Y = df_XY.Label
    X = df_XY.drop(Y, axis=1)

    def create_svm(kernel):  # Support Vector Machine
        return SVC(kernel=kernel, gamma='scale')

    def create_knn(n_neighbors):  # KNeighbors Classifier
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    def create_randomforest(n_estimators):  # RandomForest Classifier
        return RandomForestClassifier(n_estimators=n_estimators,
                                      random_state=0)

    def get_models():  # Models
        models = [create_svm('linear'),
                  create_svm('rbf'),
                  create_randomforest(100),
                  create_knn(1),
                  create_knn(5),
                  create_knn(10)]
        return models

    # Function for error rates of each model

    def get_scores(models,  X_train, X_test, Y_train, Y_test):
        errors = []
        for model in models:
            model.fit(X_train, Y_train)
            Y_predicted = model.predict(X_test)
            error = ((1-accuracy_score(Y_test, Y_predicted)).round(4))
            errors.append(error)
        return (errors)
    # K Fold for 5 splits
    kf = StratifiedKFold(n_splits=5, random_state=None)
    kf.get_n_splits(X, Y)
    errors_final = []
    for i, j in kf.split(X, Y):
        X_train, X_test = X.iloc[i], X.iloc[j]
        Y_train, Y_test = Y.iloc[i], Y.iloc[j]
        models = get_models()
        errors = get_scores(models, X_train, X_test, Y_train, Y_test)
        errors_final.append(errors)
    errors_finalarr = np.array(errors_final)
    errors = errors_finalarr.mean(axis=0).round(3)
    model_names = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]
    for i in range(len(model_names)):
        print("Model name: " + model_names[i] +
              " Error Rate: " + str(errors[i]))
    # Creating a dataframe for storing error values of classifiers
    errors = np.array(errors)
    errors.transpose()
    kfolds_scores = pd.DataFrame(columns=model_names)
    kfolds_scores.loc[len(kfolds_scores)] = errors
    kfolds_scores.index = ['errors']

    # Function for json file of mnist data

    def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
        COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
        df = kfold_scores
        if not isinstance(df, DataFrame):
            raise ValueError(" `kfold_scores` must be a pandas DataFrame.")
        if kfold_scores.shape != (1, 6):
            raise ValueError("DataFrame must have 1 row and 6 columns.")
        if not np.alltrue(sorted(df.columns) == COLS):
            raise ValueError("Columns are incorrectly named.")
        if not df.index.values[0] == "errors":
            raise ValueError("bad name. Use `kfold_score.index = ['errors']`.")

        if np.min(df.values) < 0 or np.max(df.values) > 0.10:
            raise ValueError(
                "Your K-Fold error rates are too extreme.,\r\n"
                " NOT percentage error rates. ensure your DataFrame,\r\n"
                " contains error rates and not accuracies. ,\r\n"
                "there is probably something else wrong with your code..\r\n"
            )

        if df.loc["errors", "svm_linear"] > 0.07:
            raise ValueError("Your svm_linear error rate is too high.")
        if df.loc["errors", "svm_rbf"] > 0.03:
            raise ValueError("Your svm_rbf error rate is too high.")
        if df.loc["errors", "rf"] > 0.05:
            raise ValueError("Your Random Forest error rate is too high.")
        if df.loc["errors", ["knn1", "knn5", "knn10"]].min() > 0.04:
            raise ValueError("One of your KNN error rates is too high.")

        outfile = Path("__file__").resolve().parent / "kfold_mnist.json"
        df.to_json(outfile)
        print(f" MNIST data error rates successfully saved to {outfile}")
    save_mnist_kfold(kfolds_scores)


if __name__ == "__main__":
    solution()