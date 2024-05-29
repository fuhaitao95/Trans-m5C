# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:16:30 2023


"""


# %% Import

import copy
import itertools
import numpy as np
import optuna
import os
import pandas as pd
import random
import sys
import time

import torch
from torch import nn
from torch.nn import Parameter

from parameters import Args

from Datasets import GetRawData
from Datasets import SplitData

from utils.util import PreData
from utils.calc_metrics import get_metrics
# from TrainValidateTest import traValTes

# from models import TraML
from models import TraGRU
# from models import traDGA_5mC

# from GenerateRawData import RawData

# from plotDistribution import pltDisMain
# from plotAtt import plotHM_main





# %% Write result
def writeResult(args, result, prefix=None):
    if prefix is None:
        prefix = args['prefix']

    result_keys = result.keys()
    print(f'result_keys: {result_keys}')
    
    outFileLs = copy.deepcopy(args['outFileLs'])

    outName = f'{prefix}_other_info.csv'
    args['outFileLs'].append(outName)
    with open(outName, 'a') as fobj:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                fobj.write(f'{key}, {value}\n')

    if 'metricsTra_pd' in result_keys:
        metricsTra_pd = result['metricsTra_pd']
        outName = f'{prefix}_metricsTra.csv'
        args['outFileLs'].append(outName)
        metricsTra_pd.to_csv(outName, float_format='%.5f')

    if 'metricsVal' in result_keys:
        metricsVal_pd = result['metricsVal_pd']
        outName = f'{prefix}_metricsVal.csv'
        args['outFileLs'].append(outName)
        metricsVal_pd.to_csv(outName, float_format='%.5f')

    if 'metricsTes' in result_keys:
        metricsTes_pd = result['metricsTes_pd']
        outName = f'{prefix}_metricsTes.csv'
        args['outFileLs'].append(outName)
        metricsTes_pd.to_csv(outName, float_format='%.5f')

    if 'y_label_pred_pd' in result_keys:
        y_label_pred_pd = result['y_label_pred_pd']
        y_label_pred_pd_name = result['y_label_pred_pd_name']
        args['outFileLs'].append(y_label_pred_pd_name)
        y_label_pred_pd.to_csv(y_label_pred_pd_name, float_format='%.5f')
        
        tesLabel=result['y_label_pred_pd'].loc[:,'tesLabel'].values
        tesPredictProb=result['y_label_pred_pd'].loc[:,'tesPredictProb'].values
        seven_metrics = get_metrics(tesLabel,tesPredictProb)
        columns = ['AUPR', 'AUC', 'F1score', 'Accuracy', 'Recall', 'Specificity', 'Precision', 'Thresholds_max']
        result_seven = pd.DataFrame(np.array(seven_metrics).reshape(1,-1), columns=columns)

        seven_name = f'{prefix}_seven_metrics_testAUC{round(seven_metrics[1],5)}.csv'
        args['outFileLs'].append(seven_name)
        result_seven.to_csv(seven_name, float_format='%.5f')
        
    argsObj.writeParameters(args)
    args['outFileLs'] = copy.deepcopy(outFileLs)

    return



# %% experiment

class Experiment(object):
    def __init__(self, args):
        self.args = args

        result = self.runExp()
        self.result = result

        return

    def runExp(self):
        args = self.args
        expName = args['expName']
        result = dict()

        if False:
            sys.exit('Wrong!')

        elif expName == 'TraValTes':
            dataObj = GetRawData(args)
            dataDict = dataObj.dataDict

            result = self.TraValTes(dataDict)
            writeResult(args, result)

        else:
            sys.exit(f'Wrong experiment name: {expName}! ')
        return result


    def TraValTes(self, dataDict):
        args = self.args

        modelName = args['modelName']

        if False:
            sys.exit('Wrong!')
            
        elif modelName == 'Transformer':
            traGRU_obj = TraGRU(args, dataDict)
            result = traGRU_obj.traValTes()
            
        else:
            sys.exit(f'Wrong modelName: {modelName}!')
        return result


# %% main
if __name__ == '__main__':
    argsObj = Args()
    args = argsObj.args

    expName = args['expName']

    expObj = Experiment(args)
    result = expObj.result

    # argsObj.writeParameters(args)
    # print(result)


# %% End