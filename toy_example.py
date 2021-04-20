##########################################################################
## This script generates all subgroup results for bottleneck prediction ##
##########################################################################

# All library imports

# general imports
import pandas as pd
import numpy as np

#setting printing parameters so subgroup descriptions actually display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 150)

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
X = X.loc[X.isna().sum(axis=1) == 0][["Age", "Fare", "Pclass"]]
#create simple model to test PredictionQF
RF = rf_classifier.fit(X[["Age", "Fare", "Pclass"]], y)
y_hat = RF.predict_proba(X)[:,1]

target = ps.PredictionTarget(y.to_numpy(), y_hat, roc_auc_score)

searchspace = ps.create_selectors(X[["Age","Fare", "Pclass"]], ignore=['Survived'])
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

############################################################
## Toy example using the test dataset to generate answers ##
############################################################


from pysubgroup.tests.DataSets import get_credit_data
data = get_credit_data()

np.random.seed(1111)
target_variables = np.random.randint(low=0, high=2, size=1000)
target_estimates = np.random.uniform(size=1000)
target = ps.PredictionTarget(target_variables, target_estimates, roc_auc_score)

searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['credit_amount'])
searchSpace_Numeric = [] #ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
searchSpace = searchSpace_Nominal + searchSpace_Numeric

task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=5, qf=ps.CountCallsInterestingMeasure(ps.PredictionQFNumeric(1, False)))

resultBS = ps.BeamSearch().execute(task)
resultA = ps.Apriori(use_numba=False).execute(task)
resultA_numba = ps.Apriori(use_numba=True).execute(task)
resultSimpleDFS = ps.SimpleDFS().execute(task)
resultDFS = ps.DFS(ps.BitSetRepresentation).execute(task)
resultDFS.to_dataframe()
