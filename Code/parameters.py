# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:03:04 2023


"""


# %% Import
import argparse
import json
import os
import pandas as pd
import sys
import time
import torch

from utils.util import setSeed


# %% object of parameters
class Args(object):
    '''
    Input: none
    Output: object for holding predefined parameters
    Attributes: args, argparse.Namespace
    '''

    def __init__(self):
        parser_args = self.constructArgs()

        config_args = self.getConfig(parser_args)
        args = self.getArgs(parser_args, config_args)

        args = self.setDevice(args)
        args = self.createOutDir(args)
        args = self.setPrefix(args)

        args = self.writeParameters(args)

        setSeed(args['seed'])

        args['nbWords'] = 4 ** args['num_k'] + 1

        self.args = args
        return

    def constructArgs(self):
        parser = argparse.ArgumentParser(
            description='RNA modification site prediction')
        parser.add_argument('--config', default="../Config/config_general.json",
                            help='Please give a config.json file for setting hyper-parameters')
        parser.add_argument('--expName', help='experiment name')
        parser.add_argument('--dataName',
                            help='The dataset name')
        parser.add_argument('--modelName', type=str)
        parser.add_argument("--featureType", type=str)

        parser.add_argument('--sampleLen', type=int,
                            help='the length of the sample')
        parser.add_argument('--num_k', type=int,
                            help='the num_k bases for embedding')
        # parameters for deep learning-based model
        parser.add_argument('--num_layers', type=int)
        parser.add_argument('--hidDim', type=int)
        parser.add_argument('--num_linear', type=int,
                            help='number of linear layer for prediction')
        parser.add_argument('--lr', type=float,
                            help='learning rate for deep learning')
        parser.add_argument('--weight_decay',
                            type=float, help='weight decay for deep learning')
        parser.add_argument('--dropProb', type=float,
                            help='dropout probability for deep learning')
        parser.add_argument('--n_heads', type=int)
        parser.add_argument('--d_ff', type=int)
        parser.add_argument('--norm', type=str)
        parser.add_argument('--embedType', type=str)

        parser.add_argument('--embeddingDim', type=int)

        parser.add_argument('--bidirectional', type=int, help='True or False for GRU')

        parser.add_argument('--epochs', type=int,
                            help='total epochs for deep learning')
        parser.add_argument('--outputSize', type=int)
        parser.add_argument('--earlyFlag',
                            type=int, help='earlyFlag for deep learning')
        parser.add_argument('--patience',
                            type=int, help='patience for deep learning')
        parser.add_argument('--batchSize', type=int)
        parser.add_argument('--interval', type=int)
        parser.add_argument('--bestEpoch', type=int)

        parser.add_argument('--traRatio', type=float)
        parser.add_argument('--valRatio', type=float)
        parser.add_argument('--tesRatio', type=float)

        parser.add_argument('--dev', type=str)
        parser.add_argument('--seed', type=int)
        parser.add_argument('--expKey', type=str)
        parser.add_argument('--n_trials', type=int)

        args = parser.parse_args()

        return args

    def getConfig(self, parser_args):
        with open(parser_args.config) as f:
            config_args = json.load(f)
        return config_args

    # 如果有命令行的值，就用命令行的值。否则就用config_args.json的设置
    def getArgs(self, parser_args, config_args):
        args = dict()
        for key, value in vars(parser_args).items():
            if value is not None:
                args[key] = value
            else:
                args[key] = config_args[key]['default']
                
        args['bidirectional'] = bool(args['bidirectional'])
        args['earlyFlag'] = bool(args['earlyFlag'])
        dataName = args['dataName']
        args['species'] = species = dataName.split('_')[1]
        assert species in ['Human', 'Mouse']
        args['dataType'] = dataType = dataName.split('_')[2]
        assert dataType in ['exon', 'transcript']
        # args['ctcca'] = ctcca = dataName.split('_')[3]
        # assert ctcca in ['Y', 'N', 'YN']
        return args

    def setDevice(self, args):
        dev = args['dev']
        modelName = args['modelName']
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if modelName == 'Jia_DGA_5mC':
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
            args['dev'] = dev = 'cpu'
            print('cuda is not available')
        elif torch.cuda.is_available() and dev.startswith('cuda'):
            # gpu_id = int(dev[-1])
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f'cuda is available with GPU: {dev}')
        else:
            args['dev'] = dev = 'cpu'
            print('cuda os not available')
        args['device'] = torch.device(dev)
        return args

    def createOutDir(self, args):
        dataName = args['dataName']
        expName = args['expName']
        args['outPath'] = outPath = f'../Output/{dataName}/{expName}'
        if not os.path.exists(outPath):
            os.makedirs(outPath)

        seed = args['seed']
        args['datapklName'] = f'{outPath}/dataset_seed{seed}.pkl'
        return args

    def setPrefix(self, args):
        outPath = args['outPath']
        modelName = args['modelName']
        expKey = args['expKey']
        timeStr = time.strftime("%Y%m%d_%H%M%S_", time.localtime())[2:]
        args['prefix'] = (f'{outPath}/{timeStr}_{modelName}_{expKey}')
        args['outFileLs'] = []
        return args

    def writeParameters(self, args):
        prefix = args['prefix']

        para = pd.DataFrame(args.items(), columns=['parameter', 'value'])
        output_name = prefix + '_parameters.csv'
        args['outFileLs'].append(output_name)
        para.to_csv(output_name, sep=',', mode='a', index=True, header=True)
        return args


# %% Main
# python parameters.py --expName TraValTes
if __name__ == '__main__':
    argsObj = Args()
    args = argsObj.args
    print(args)
    pass

# %% End
