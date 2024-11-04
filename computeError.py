import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
        f1_score    -- float, test "micro" averaged f1 score
    """

    # compute cross-validation error using StratifiedShuffleSplit over ntrials
    # uses StratifiedShuffleSplit (be careful of the parameters)
    train_error = np.zeros(ntrials)
    test_error  = np.zeros(ntrials)
    fpr         = np.zeros(ntrials)
    fnr         = np.zeros(ntrials)
    f1_score    = np.zeros(ntrials)

    sss=StratifiedShuffleSplit(n_splits=ntrials, test_size=test_size, random_state=0)

    for i, (train_index, test_index) in enumerate(sss.split(X,y)):
      clf.fit(X[train_index,:], y[train_index])

      y_pred_train = clf.predict(X[train_index])
      train_error[i] = 1 - metrics.accuracy_score(y[train_index], y_pred_train, normalize=True)

      y_pred_test = clf.predict(X[test_index])
      test_error[i] = 1 - metrics.accuracy_score(y[test_index], y_pred_test, normalize=True)

      f1_score[i] = metrics.f1_score(y[test_index], y_pred_test)
      fp = np.sum((y_pred_test == 1) & (y[test_index] == 0))
      tp = np.sum((y_pred_test == 1) & (y[test_index] == 1))
      fn = np.sum((y_pred_test == 0) & (y[test_index] == 1))
      tn = np.sum((y_pred_test == 0) & (y[test_index] == 0))
      if (fp + tn) > 0:
         fpr[i] = fp / (fp + tn)
      else:
         fpr[i] = 0
      if (fn + tp) > 0:
         fnr[i] = fn / (fn + tp) 
      else:
         fnr[i] = 0

    train_error = np.mean(train_error)
    test_error = np.mean(test_error)
    f1_score = np.mean(f1_score)
    fpr      = np.mean(fpr)
    fnr      = np.mean(fnr)
    return train_error, test_error, f1_score, fpr, fnr



