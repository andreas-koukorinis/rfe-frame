from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("nci60.csv")
# X = np.array(df.iloc[:,2:])
# y = np.array(df.iloc[:,1])
X = df.iloc[:,2:]
y = df.iloc[:,1]

# Define the rfe-svm function
def rfe_svm_tim(X, y, num_feature_to_selected):
    import time
    clf = SVR()
    model = clf.set_params(kernel='linear')
    X_new = X
    record = {}
    keep_ = X.shape[1]-num_feature_to_selected
    start_time = time.time()
    for i in range(keep_):
        # fit the svm and get the weights of features
        fit = model.fit(X_new, y)
        # remove the feature with the smallest weight
        X_new = X_new.drop(X_new.columns[[np.argmin(fit.coef_)]], 1)
        # record the index and value of the deleted feature
        record.update({str(np.argmin(fit.coef_)):fit.coef_[0,np.argmin(fit.coef_)]})
    elapsed_time = time.time() - start_time
    return X_new, record, elapsed_time

# Do the feature selecltion and get the selected features
X_new, record, elapsed_time = rfe_svm_tim(X, y, 25)
# for key,value in record.items():
#     print(key,':',value)
len(record)
X_new.shape[1]
elapsed_time

# Do the cross-validation
from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=3)
clf = SVR()
model = clf.set_params(kernel='linear')
score = cross_val_score(model, X_new, y, cv=k_fold, n_jobs=-1)
np.mean(score)