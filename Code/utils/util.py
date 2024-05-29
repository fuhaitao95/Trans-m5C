# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:50:01 2021


"""


import itertools
import numpy as np
import os
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import sys

import torch
from torch.utils.data import Dataset


def setSeed(seed):
    os.environ['PYTHONASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

def mRNA2DNA(seqData):
    seqData = [sample.replace('a', 't').replace('A', 'T') for sample in seqData]
    seqData = [sample.replace('u', 'a').replace('U', 'A') for sample in seqData]
    seqData = [sample.replace('c', 'x').replace('C', 'X') for sample in seqData]
    seqData = [sample.replace('g', 'c').replace('G', 'C') for sample in seqData]
    seqData = [sample.replace('x', 'g').replace('X', 'G') for sample in seqData]
    seqReverse = seqData[::-1]
    return seqReverse
def tRNA2DNA(seqData):
    seqData = [sample.replace('u', 't').replace('U', 'T') for sample in seqData]
    return seqData

def skMetrics(y_label, y_pred):
    # AUPR AUC
    AUPR = round(average_precision_score(y_label, y_pred), 4)
    AUC = round(roc_auc_score(y_label, y_pred), 4)
    metrics = np.array([AUPR, AUC, 3, 4, 5, 6, 7])
    return metrics


class PreData(object):
    def __init__(self, args, dataDict):
        self.args = args
        dataDict['XTra'] = self.str2num(dataDict['seqTra'])
        dataDict['XVal'] = self.str2num(dataDict['seqVal'])
        dataDict['XTes'] = self.str2num(dataDict['seqTes'])
        return

    def getPermute(self):
        kMer = self.args['kMer']
        f = ['a', 'c', 'g', 't']
        c = itertools.product(*([f for i in range(kMer)]))

        ind2kmer = []
        kmer2ind = dict()
        ind2kmer.append('null')
        kmer2ind['null'] = 0
        for i, value in enumerate(c):
            temp = ''.join(value)
            ind2kmer.append(temp)
            kmer2ind[temp] = i+1

        return kmer2ind, ind2kmer

    def str2num(self, data):
        kMer = self.args['kMer']
        kmer2ind, ind2kmer = self.getPermute()

        dataNum = []
        iSample = 0
        for iSample in range(len(data)):
            sample = data[iSample]
            # assert len(sample) == length
            sampleNum = []
            for jSite in range(len(sample)-kMer+1):
                kMer_value = sample[jSite: jSite + kMer]
                if 'n' in kMer_value.lower():
                    sampleNum.append(0)
                else:
                    sampleNum.append(kmer2ind[kMer_value])
            dataNum.append(sampleNum)
        dataNum = np.array(dataNum)
        return dataNum


class DealDataset(Dataset):
    def __init__(self, X, Y):
        self.Y = Y
        self.x_data = torch.from_numpy(X).long()
        self.y_data = torch.from_numpy(Y).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.Y)

class torchDataset(Dataset):
    def __init__(self, X, Y):
        self.Y = Y
        self.x_data = torch.from_numpy(X).long()
        self.y_data = torch.from_numpy(Y).float()
        return

    def __getitem__(self, index):
        x_i, y_i = self.x_data[index], self.y_data[index]
        return x_i, y_i

    def __len__(self):
        length = len(self.Y)
        return length


def processZhang2020iPromoter_5mCNegative(prefix=''):
    # prefix = '../'
    fileDir = '../Data/Zhang2020iPromoter_5mC/'
    inName = 'all_negative.fasta'
    outName = 'equal_negative.fasta'
    length = 69750

    fileName = f'{prefix}{fileDir}{inName}'
    if not os.path.exists(fileName):
        sys.exit(f'Wrong path: {fileName}')
    dataNote = open(fileName, 'r').readlines()[0::2]
    dataNeg = open(fileName, 'r').readlines()[1::2]

    random.seed(0)
    index = random.choices(range(len(dataNeg)), k=length)

    fileName = f'{prefix}{fileDir}{outName}'
    with open(fileName, 'w') as fobj:
        for ind in index:
            fobj.write(dataNote[ind].strip()+f'_{ind}\n')
            fobj.write(dataNeg[ind].strip()+'\n')
    return
