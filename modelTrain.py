import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datetime import datetime
from computeError import error

def preprocess(csv):
    # read in data
    data = pd.read_csv(csv, encoding='utf-8')

    # relabel
    data.rename(columns={'menu': 'label', 'content': 'text'}, inplace=True)
    data['label'] = data['label'].map({'N': 0, 'Y': 1})

    # remove any duplicates
    data.drop_duplicates(keep='first', inplace = True)

    # vectorize feature space
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])

    # reduce dimensionality of X input
    targetVariance = 0.99
    testComponents = range(50, 1001, 50)
    explainedVar   = []
    # loop over to find explained variance for components ranging from 50 to 1000
    best_k = None
    for idx, k in enumerate(testComponents):
        svd = TruncatedSVD(n_components=k, random_state=42)
        svd.fit(X)
        variance = svd.explained_variance_ratio_.sum()
        explainedVar.append(variance)
        # get the first n_component that explains for at least 99% of variance 
        if variance >= targetVariance: 
            best_k = testComponents[idx]
            break 
    joblib.dump(vectorizer, 'modelBuilder/vectors.joblib')
    # fit X with singular value decomp 
    svd = TruncatedSVD(n_components=best_k, random_state=42)
    X = svd.fit_transform(X)
    joblib.dump(svd, 'modelBuilder/svd.joblib')
    # y label to numpy array for ML 
    y = data['label'].to_numpy()

    return X, y, best_k



def getModel(X, y):

    lambda_ = [0.01, 0.1, 1, 10, 100]
    fpr_s = []
    for param in lambda_:
        clf = LogisticRegression(C=1/param, random_state=42, max_iter=1000)
        _, _, _, fpr,_ = error(clf, X, y)
        fpr_s.append(fpr)
    minError_idx = np.argmin(fpr_s)
    bestReg      = lambda_[minError_idx]
    model        = LogisticRegression(C=1/bestReg, random_state=42, max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model.fit(X_train, y_train)
    score  = model.score(X_test, y_test)

    return X_test, y_test, model, score, bestReg

def chooseThresh(model, X_test, y_test):
    class_probs = model.predict_proba(X_test)[:,1]
    thresholds  = np.arange(0.5, 0.71, 0.05)
    fpr_values  = []

    for thresh in thresholds:
        predicted_classes = (class_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_classes).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # avoid division by zero
        fpr_values.append(fpr)

    minFPR_idx = np.argmin(fpr_values)
    best_threshold = thresholds[minFPR_idx]
    lowestFPR = fpr_values[minFPR_idx]

    return best_threshold, lowestFPR


def trainModel():
    X, y, best_k = preprocess(csv='allMenus.csv')
    X_test, y_test, model, score, bestReg = getModel(X, y)

    threshold, fpr = chooseThresh(model, X_test, y_test)

    with open("modelBuilder/modelInfo.txt", "a") as file:
        today = datetime.today().date()
        file.write(f"Date: {today}\n")
        file.write(f'Accuracy: {score}\n')
        file.write(f'Threshold: {threshold}\n')
        file.write(f'false positive rate: {fpr}\n')
        file.write(f'Number of Components: {best_k}\n')
        file.write(f'Regularizer: {best_k}\n')
        file.write("==============================\n")

    return joblib.dump(model, 'modelBuilder/model.joblib')

trainModel()











