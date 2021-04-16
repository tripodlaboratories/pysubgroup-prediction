##########################################################################
## This script generates all subgroup results for bottleneck prediction ##
##########################################################################

# All library imports

# general imports
import pandas as pd
import numpy as np

# subgroup analysis package
import pysubgroup as ps

# sklearn imports for evaluating results
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc #collects AUPRC + AUROC
from sklearn.metrics import confusion_matrix # used to collect true-pos. fals-negs, ect.

#test model
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(max_depth=4, random_state=420)
# Load the example dataset
from pysubgroup.tests.DataSets import get_titanic_data
data = get_titanic_data()

select = data.columns[data.dtypes.transform(lambda x: True if x in ["float64", "int64"] else False)]
X = data[set(select) - set(["Survived"])]
y = data["Survived"]

y = y.loc[X.isna().sum(axis=1) == 0]
X = X.loc[X.isna().sum(axis=1) == 0]
#create simple model to test PredictionQF
RF = rf_classifier.fit(X, y)
y_hat = RF.predict_proba(X)[:,1]

target = ps.PredictionTarget(y.to_numpy(), y_hat, roc_auc_score)

searchspace = ps.create_selectors(X, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask (
    X,
    target,
    searchspace,
    result_set_size=5,
    depth=2,
    qf=ps.PredictionQFNumeric(a=0.5))

result = ps.BeamSearch().execute(task)

result.to_dataframe()
roc_auc_score(y[X.Age >= 38.0], y_hat[X.Age >= 38.0])




