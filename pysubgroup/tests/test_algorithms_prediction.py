# 383476.7679999999:      own_telephone=='b'yes''
# 361710.05800000014:     foreign_worker=='b'yes'' and own_telephone=='b'yes''
# 345352.9920000001:      other_parties=='b'none'' and own_telephone=='b'yes''
# 338205.08:      foreign_worker=='b'yes'' and own_telephone=='b'yes'' and personal_status=='b'male single''
# 336857.8220000001:      own_telephone=='b'yes'' and personal_status=='b'male single''
# 323586.28200000006:     foreign_worker=='b'yes'' and other_parties=='b'none'' and own_telephone=='b'yes''
# 320306.81600000005:     job=='b'high qualif/self emp/mgmt''
# 300963.84599999996:     class=='b'bad'' and own_telephone=='b'yes''
# 299447.332:     foreign_worker=='b'yes'' and job=='b'high qualif/self emp/mgmt''
# 297422.98200000013:     foreign_worker=='b'yes'' and other_parties=='b'none'' and own_telephone=='b'yes'' and personal_status=='b'male single''

import unittest
import pysubgroup as ps
import numpy as np
from sklearn.metrics import roc_auc_score

from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase

class TestPredictionQFNumeric(unittest.TestCase):
    def test_constructor(self):
        ps.PredictionQFNumeric(0)
        ps.PredictionQFNumeric(1.0)
        ps.PredictionQFNumeric(0, invert=True)
        ps.PredictionQFNumeric(0, invert=False)
        with self.assertRaises(ValueError):
            ps.PredictionQFNumeric('test')


class TestAlgorithmsWithNumericTarget(TestAlgorithmsBase, unittest.TestCase):
    def setUp(self):
        #NS_all = ps.EqualitySelector(True)
        NS_payment = ps.EqualitySelector("other_payment_plans",b"none")
        NS_foreign_worker = ps.EqualitySelector("foreign_worker", b"yes")
        NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        NS_housing = ps.EqualitySelector("housing", b'own')
        NS_class = ps.EqualitySelector("class", b"good")
        DFSo = [[NS_foreign_worker],
                [NS_other_parties],
                [NS_foreign_worker, NS_other_parties],
                [NS_payment],
                [NS_foreign_worker, NS_payment],
                [NS_other_parties, NS_payment],
                [NS_housing],
                [NS_class],
                [NS_foreign_worker, NS_other_parties, NS_payment]]
        self.DFSresult = list(map(ps.Conjunction, DFSo))
        self.DFSresult.insert(0,True)
        self.DFSqualities = [500.4980179286455,
                483.3153195123844,
                459.2862838915471,
                444.60343785358896,
                398.25539855072464,
                384.0460358056267,
                362.090608537693,
                355.0749649843413,
                355.010575658835,
                349.8188702669149]
        o = [[NS_foreign_worker],
                [NS_other_parties],
                [NS_foreign_worker, NS_other_parties],
                [NS_payment],
                [NS_foreign_worker, NS_payment],
                [NS_other_parties, NS_payment],
                [NS_housing],
                [NS_class],
                [NS_foreign_worker, NS_other_parties, NS_payment],
                [NS_foreign_worker, NS_housing]]
        self.result = list(map(ps.Conjunction, o))
        self.qualities = [483.3153195123844,
                459.2862838915471,
                444.60343785358896,
                398.25539855072464,
                384.0460358056267,
                362.090608537693,
                355.0749649843413,
                355.010575658835,
                349.8188702669149,
                342.20780439530444]
        np.random.seed(1111)
        self.target_variables = np.random.randint(low=0, high=2, size=1000)
        self.target_estimates = np.random.uniform(size=1000)
        data = get_credit_data()
        target = ps.PredictionTarget(self.target_variables, self.target_estimates, roc_auc_score)
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['credit_amount'])
        searchSpace_Numeric = [] #ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=5, qf=ps.CountCallsInterestingMeasure(ps.PredictionQFNumeric(1, False)))
        #
    def test_SimpleDFS(self):
        self.runAlgorithm(ps.SimpleDFS(), "SimpleDFS", self.DFSresult, self.DFSqualities, self.task)
    #
    def test_DFS(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.PredictionQFNumeric(self.task.qf.a, False))
        self.runAlgorithm(ps.DFS(ps.BitSetRepresentation), "DFS", self.DFSresult, self.DFSqualities, self.task)
    #
    def test_BeamSearch(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.PredictionQFNumeric(self.task.qf.a, False))
        self.runAlgorithm(ps.BeamSearch(), "BeamSearch sum", self.result, self.qualities, self.task)
    #
    def test_Apriori_no_numba(self):
        self.runAlgorithm(ps.Apriori(use_numba=False), "Apriori use_numba=False", self.result, self.qualities, self.task)
    #
    def test_Apriori_with_numba(self):
        self.runAlgorithm(ps.Apriori(use_numba=True), "Apriori use_numba=True", self.result, self.qualities, self.task)
    #
    #def test_SimpleSearch(self):
    #   self.runAlgorithm(ps.SimpleSearch(), "SimpleSearch", self.result, self.qualities, self.task)


if __name__ == '__main__':
    unittest.main()


   # 639577.0460000001:   duration>=30.0
   # 624424.3040000001:   duration>=30.0 AND foreign_worker=='b'yes''
   # 579219.206:  duration>=30.0 AND other_parties=='b'none''
   # 564066.4640000002:   duration>=30.0 AND foreign_worker=='b'yes'' AND other_parties=='b'none''
   # 547252.302:  duration>=30.0 AND num_dependents==1.0
   # 532099.56:   duration>=30.0 AND foreign_worker=='b'yes'' AND num_dependents==1.0
   # 491104.688:  duration>=30.0 AND num_dependents==1.0 AND other_parties=='b'none''
   # 490633.1400000001:   duration>=30.0 AND foreign_worker=='b'yes'' AND other_payment_plans=='b'none''
   # 490633.1400000001:   duration>=30.0 AND other_payment_plans=='b'none''
   # 475951.94600000005:  duration>=30.0 AND foreign_worker=='b'yes'' AND num_dependents==1.0 AND other_parties=='b'none''
