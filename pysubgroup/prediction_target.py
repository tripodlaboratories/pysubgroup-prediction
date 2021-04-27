'''
Created on 14.04.2021

@author: Tony Culos
'''
import numbers
import operator
from collections import namedtuple
from functools import total_ordering
import numpy as np
import pysubgroup as ps
import sklearn.metrics as metrics #TODO: import of entire package may be uneccesary, see if attribute list can be used


@total_ordering
class PredictionTarget:
    statistic_types = ('size_sg', 'size_dataset', 'pos_sg', 'pos_dataset', 'neg_sg', 'neg_dataset', "metric_sg", "metric_dataset")

    def __init__(self, target_variable, target_estimate, eval_func=None, eval_dict=None):
        self.target_variable = target_variable
        self.target_estimate = target_estimate
        self.eval_dict = eval_dict
        if not eval_dict is None:
            PredictionTarget.statistic_types = PredictionTarget.statistic_types + tuple([x +"_sg" for x in eval_dict.keys()]) + tuple([x +"_dataset" for x in eval_dict.keys()])
        if eval_func is None:
            self.evaluation_metric = self.default_evaluation_metric
        elif not hasattr(metrics, eval_func.__name__):
            raise ValueError("eval_func passed must be from sklearn.metrics")
        else:
            # TODO: move evaluation metric to qualit function
            self.evaluation_metric = eval_func

    def __repr__(self):
        return "T: " + str(self.target_variable) + "\nT_hat: " +str(self.target_estimate)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [self.target_variable, self.target_estimate]

    #default eval function is average sub ranking loss, see Duivesteijn & Thaele
    def default_evaluation_metric(self, y_true, y_pred):
        sorted_true = y_true[np.argsort(y_pred)]
        numerator_sum = 0
        for i in range(len(y_true)):
            if sorted_true[i] == 1: numerator_sum += (sorted_true[0:i] == 0).sum()
        return numerator_sum/y_true.sum()

    #TODO: not sure if necessary but updated to return new statistics
    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup)
        size_dataset = data.shape[0]
        metric_sg = self.evaluation_metric(self.target_variable[cover_arr], self.target_estimate[cover_arr])
        metric_dataset = self.evaluation_metric(self.target_variable, self.target_estimate)
        return (size_sg, size_dataset, metric_sg, metric_dataset)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        if cached_statistics is None or not isinstance(cached_statistics, dict):
            statistics = dict()
        elif all(k in cached_statistics for k in PredictionTarget.statistic_types):
            return cached_statistics
        else:
            statistics = cached_statistics

        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)

        statistics['size_sg'] = size_sg
        statistics['size_dataset'] = data.shape[0]

        statistics['pos_sg'] = self.target_variable[cover_arr].sum()
        statistics['pos_dataset'] = self.target_variable.sum()
        statistics['neg_sg'] = (1 - self.target_variable[cover_arr]).sum()
        statistics['neg_dataset'] = (1 - self.target_variable).sum()

        statistics['metric_sg'] = self.evaluation_metric(self.target_variable[cover_arr], self.target_estimate[cover_arr])
        statistics['metric_dataset'] = self.evaluation_metric(self.target_variable, self.target_estimate)

        if not self.eval_dict is None:
            for key in self.eval_dict.keys():
                statistics[key+"_sg"] = self.eval_dict[key](self.target_variable[cover_arr], self.target_estimate[cover_arr])
                statistics[key+"_dataset"] = self.eval_dict[key](self.target_variable, self.target_estimate)

        return statistics


class PredictionQFNumeric(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('PredictionQFNumeric_parameters', ('size_sg', 'metric_sg', 'estimate'))
    @staticmethod
    def prediction_qf_numeric(a, size_sg, metric_sg):
        return size_sg ** a * (metric_sg)

    def __init__(self, a, invert=False):
        if not isinstance(a, numbers.Number):
            raise ValueError(f'a is not a number. Received a={a}')
        self.a = a
        self.size=None
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'metric_sg')
        self.dataset_statistics = None
        self.all_target_variable = None
        self.all_target_estimate = None
        self.all_target_metric = None
        self.has_constant_statistics = False
        self.estimator = PredictionQFNumeric.OptimisticEstimator(self)

    #TODO: raise error when data does not align with target vars
    def calculate_constant_statistics(self, data, target):
        self.size = len(data)
        self.all_target_variable = target.target_variable
        self.all_target_estimate = target.target_estimate
        self.all_target_metric = target.evaluation_metric(self.all_target_variable, self.all_target_estimate)
        self.has_constant_statistics = True
        estimate = self.estimator.get_estimate(self.size, self.a)
        self.dataset_statistics = PredictionQFNumeric.tpl(self.size, self.all_target_metric, estimate)


    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        #dataset = self.dataset_statistics #can be used to compare all data AUC to subgroup AUC
        return PredictionQFNumeric.prediction_qf_numeric(self.a, statistics.size_sg, statistics.metric_sg)


    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_variable), data)
        if sg_size > 0 and np.std(self.all_target_variable[cover_arr]) != 0:
            sg_target_variable = self.all_target_variable[cover_arr]
            sg_target_estimate = self.all_target_estimate[cover_arr]
            estimate = self.estimator.get_estimate(sg_size, self.a)
            metric_sg = target.evaluation_metric(sg_target_variable, sg_target_estimate)
        else:
            estimate = float('-inf')
            metric_sg = 0# float('-inf')
        return PredictionQFNumeric.tpl(sg_size, metric_sg, estimate)


    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate

    class OptimisticEstimator:
        def __init__(self, qf):
            self.qf = qf
            self.metric = None

        def get_data(self, data):
            return data

        def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
            self.metric = float('inf')# target.evaluation_metric

        def get_estimate(self, size_sg, a):  # pylint: disable=unused-argument
            max_possible = 1
            return size_sg ** a * (max_possible) #TODO: how to extract max from all sklearn metrics dynamically
            #return self.metric(y_true=sg_target_variable, y_pred=sg_target_estimate)


# TODO Update to new format
#class GAPredictionQFNumeric(ps.AbstractInterestingnessMeasure):
#    def __init__(self, a, invert=False):
#        self.a = a
#        self.invert = invert
#
#    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
#        (instances_dataset, _, instances_subgroup, mean_sg) = subgroup.get_base_statistics(data, weighting_attribute)
#        if instances_subgroup in (0, instances_dataset):
#            return 0
#        max_mean = get_max_generalization_mean(data, subgroup, weighting_attribute)
#        relative_size = (instances_subgroup / instances_dataset)
#        return ps.conditional_invert(relative_size ** self.a * (mean_sg - max_mean), self.invert)

#    def supports_weights(self):
#        return True

#    def is_applicable(self, subgroup):
#        return isinstance(subgroup.target, NumericTarget)


#def get_max_generalization_mean(data, subgroup, weighting_attribute=None):
#    selectors = subgroup.subgroup_description.selectors
#    generalizations = ps.powerset(selectors)
#    max_mean = 0
#    for sels in generalizations:
#        sg = ps.Subgroup(subgroup.target, ps.Conjunction(list(sels)))
#        mean_sg = sg.get_base_statistics(data, weighting_attribute)[3]
#        max_mean = max(max_mean, mean_sg)
#    return max_mean
