"""Classical RBF-SVM baseline for the fraud track (AUC)."""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def rbf_svm_auc(X, y, seed: int = 0) -> float:
    Xtr, Xte, ytr, yte = train_test_split(
        np.asarray(X), np.asarray(y), test_size=0.3, random_state=seed, stratify=y
    )
    clf = SVC(kernel="rbf", probability=True, random_state=seed).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba))
