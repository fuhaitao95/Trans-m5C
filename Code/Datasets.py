#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 21:40:53 2023

@author: fht
"""


# %% Import
import copy
import numpy as np
import os
import random
import sys

from parameters import Args
from utils.util import tRNA2DNA


# %% GetRawData
class GetRawData(object):
    # in: args['dataName]
    def __init__(self, args):
        self.args = args
        dataName = args['dataName']

        if False:
            sys.exit()
        elif dataName == 'Lin2022Evaluation_Hsapiens':
            dataSeq, dataY = self.getData_Lin2022Evaluation_Hsapiens()

        elif dataName == 'Lin2022Evaluation_Mmusculus':
            dataSeq, dataY = self.getData_Lin2022Evaluation_Mmusculus()

        elif dataName == 'Lin2022Evaluation_Scerevisiae':
            dataSeq, dataY = self.getData_Lin2022Evaluation_Scerevisiae()


        elif dataName == 'Lin2022Evaluation_Athaliana':
            dataSeq, dataY = self.getData_Lin2022Evaluation_Athaliana()

        elif dataName == 'Lv2020Evaluation_Athaliana':
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.processData_Lv2020Evaluation_Athaliana()
        
        elif dataName == 'Liu2022Developmental_Human_exon_Y_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_exon/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_025914'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Human_exon_N_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_exon/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_025914'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Human_exon_YN_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_exon/generateData'
            sampleLen = args['sampleLen']            
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_025914'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
            
        elif dataName == 'Liu2022Developmental_Human_transcript_Y_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_transcript/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030030'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Human_transcript_N_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_transcript/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030030'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Human_transcript_YN_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_transcript/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030030'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
            
        elif dataName == 'Liu2022Developmental_Mouse_exon_Y_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_exon/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030134'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Mouse_exon_N_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_exon/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030134'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Mouse_exon_YN_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_exon/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030134'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
            
        elif dataName == 'Liu2022Developmental_Mouse_transcript_Y_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_transcript/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030223'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_Y_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Mouse_transcript_N_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_transcript/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030223'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_N_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Mouse_transcript_YN_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_transcript/generateData'
            sampleLen = args['sampleLen']
            if False:
                sys.exit('wrong!')
            elif sampleLen == 41:
                fileMid = '230911_030223'
            else:
                sys.exit(f'wrong time string: {fileMid}')
            filePos = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}__Transformer__dataset_{sampleLen}_YN_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
            
        elif dataName == 'Liu2022Developmental_Human_exon_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_exon_ctcca/generateData'
            fileMid = '230729_072813'
            filePos = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Human_exon_ctcca_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Human_exon_ctcca_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)

        elif dataName == 'Liu2022Developmental_Human_transcript_all':
            filePrefix = '../Output/Liu2022Developmental_Human_transcript_all/generateData'
            fileMid = '230729_074052'
            filePos = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Human_transcript_all_all_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Human_transcript_all_all_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Human_transcript_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Human_transcript_ctcca/generateData'
            fileMid = '230729_075149'
            filePos = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Human_transcript_ctcca_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Human_transcript_ctcca_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)

        elif dataName == 'Liu2022Developmental_Mouse_exon_all':
            filePrefix = '../Output/Liu2022Developmental_Mouse_exon_all/generateData'
            fileMid = '230729_081743'
            filePos = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_exon_all_all_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_exon_all_all_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Mouse_exon_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_exon_ctcca/generateData'
            fileMid = '230729_081825'
            filePos = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_exon_ctcca_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_exon_ctcca_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)

        elif dataName == 'Liu2022Developmental_Mouse_transcript_all':
            filePrefix = '../Output/Liu2022Developmental_Mouse_transcript_all/generateData'
            fileMid = '230729_082037'
            filePos = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_transcript_all_all_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_transcript_all_all_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)
        elif dataName == 'Liu2022Developmental_Mouse_transcript_ctcca':
            filePrefix = '../Output/Liu2022Developmental_Mouse_transcript_ctcca/generateData'
            fileMid = '230729_082155'
            filePos = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_transcript_ctcca_ctcca_pos.csv'
            fileNeg = f'{filePrefix}/{fileMid}_generateData_Liu2022Developmental_Mouse_transcript_ctcca_ctcca_neg.csv'
            dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes = self.getData_Liu2022Developmental(filePos, fileNeg)

        elif dataName == 'Chen2020m5CPredSVM_A':
            dataSeq, dataY = self.getData_Chen2020m5CPredSVM_A()
        elif dataName == 'Chen2020m5CPredSVM_H':
            dataSeq, dataY = self.getData_Chen2020m5CPredSVM_H()
        elif dataName == 'Chen2020m5CPredSVM_M':
            dataSeq, dataY = self.getData_Chen2020m5CPredSVM_M()

        elif dataName == 'Zhang2020iPromoter_5mC':
            dataSeq, dataY = self.getData_Zhang2020iPromoter_5mC()

        else:
            sys.exit(f'Wrong data name: {dataName}')

        self.dataSeq, self.dataY = dataSeq, dataY
        dataDict = dict()
        dataDict['dataSeq'] = [temp.strip().upper().replace('U', 'T') for temp in dataSeq]
        dataDict['dataY'] = dataY
        dataDict['seqTra'] = [temp.strip().upper().replace('U', 'T') for temp in seqTra]
        dataDict['seqVal'] = [temp.strip().upper().replace('U', 'T') for temp in seqVal]
        dataDict['seqTes'] = [temp.strip().upper().replace('U', 'T') for temp in seqTes]
        dataDict['yTra'] = yTra
        dataDict['yVal'] = yVal
        dataDict['yTes'] = yTes
        
        path = f'../Output/{dataName}/saveTraValTes'
        if not os.path.exists(path):
            os.makedirs(path)
        filePrefix = f'{path}/tra_seq'
        saveSequences(dataDict['seqTra'], yTra, filePrefix)
        filePrefix = f'{path}/val_seq'
        saveSequences(dataDict['seqVal'], yVal, filePrefix)
        filePrefix = f'{path}/tes_seq'
        saveSequences(dataDict['seqTes'], yTes, filePrefix)
        
        self.dataDict = dataDict
        return

    def splitTraValTes(self, dataSeq, dataY):
        # seed = 0
        # traRatio = 0.7
        # valRatio = 0.1
        # tesRatio = 0.2
        args = self.args
        seed = args['seed']
        traRatio = args['traRatio']
        valRatio = args['valRatio']
        tesRatio = args['tesRatio']

        indexPos = np.nonzero(dataY)[0]
        indexNeg = np.where(dataY == 0)[0]
        assert len(indexPos) + len(indexNeg) == len(dataY)

        random.seed(seed)
        random.shuffle(indexPos)
        random.seed(seed+1)
        random.shuffle(indexNeg)

        traNumPos = int(len(indexPos)*traRatio)
        traNumNeg = int(len(indexNeg)*traRatio)
        valNumPos = int(len(indexPos)*valRatio)
        valNumNeg = int(len(indexNeg)*valRatio)
        tesNumPos = int(len(indexPos)*tesRatio)
        tesNumNeg = int(len(indexNeg)*tesRatio)

        indexTra = np.hstack((indexPos[: traNumPos], indexNeg[: traNumNeg]))
        indexVal = np.hstack((indexPos[traNumPos: traNumPos + valNumPos],
                              indexNeg[traNumNeg: traNumNeg + valNumNeg]))
        indexTes = np.hstack((indexPos[traNumPos + valNumPos:],
                              indexNeg[traNumNeg + valNumNeg:]))

        seqTra = [dataSeq[ind] for ind in indexTra]
        seqVal = [dataSeq[ind] for ind in indexVal]
        seqTes = [dataSeq[ind] for ind in indexTes]

        yTra = dataY[np.array(indexTra)]
        yVal = dataY[np.array(indexVal)]
        yTes = dataY[np.array(indexTes)]

        return seqTra, seqVal, seqTes, yTra, yVal, yTes


    def getData_Lin2022Evaluation_Hsapiens(self):

        fileName = '../Data/data_Lin2022Evaluation/human/human.fasta'
        dataSeq = open(fileName, 'r').readlines()[1::2]

        dataSeq = [sample.upper().strip() for sample in dataSeq]

        dataSeq = tRNA2DNA(dataSeq)
        dataY = np.array([1] * 120 + [0] * 120)

        return dataSeq, dataY

    def getData_Lin2022Evaluation_Mmusculus(self):

        fileName = '../Data/data_Lin2022Evaluation/mouse/mouse.fasta'
        dataSeq = open(fileName, 'r').readlines()[1::2]

        dataSeq = [sample.upper().strip() for sample in dataSeq]

        dataSeq = tRNA2DNA(dataSeq)
        dataY = np.array([1] * 97 + [0] * 97)

        return dataSeq, dataY


    def getData_Lin2022Evaluation_Scerevisiae(self):

        fileName = '../Data/data_Lin2022Evaluation/S.cerevisiae/S.cerevisiae.fasta'
        dataSeq = open(fileName, 'r').readlines()[1::2]

        dataSeq = [sample.upper().strip() for sample in dataSeq]

        dataSeq = tRNA2DNA(dataSeq)
        dataY = np.array([1] * 211 + [0] * 211)

        return dataSeq, dataY


    def getData_Lin2022Evaluation_Athaliana(self):

        fileTraPos = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana5289_pos.fasta'
        dataTraPos = open(fileTraPos, 'r').readlines()[1::2]
        fileIndPos = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana1000indep_pos.fasta'
        dataIndPos = open(fileIndPos, 'r').readlines()[1::2]
        dataPos = dataTraPos + dataIndPos

        fileTraNeg = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana5289_neg.fasta'
        dataTraNeg = open(fileTraNeg, 'r').readlines()[1::2]
        fileIndNeg = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana1000indep_neg.fasta'
        dataIndNeg = open(fileIndNeg, 'r').readlines()[1::2]
        dataNeg = dataTraNeg + dataIndNeg

        dataSeq = dataPos + dataNeg
        dataSeq = [sample.upper().strip() for sample in dataSeq]

        dataSeq = tRNA2DNA(dataSeq)
        dataY = np.array([1] * len(dataPos) + [0] * len(dataNeg))

        return dataSeq, dataY

    def processData_Lv2020Evaluation_Athaliana(self):
        fileTraPos = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana5289_pos.fasta'
        dataTraPos = open(fileTraPos, 'r').readlines()[1::2]
        fileIndPos = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana1000indep_pos.fasta'
        dataIndPos = open(fileIndPos, 'r').readlines()[1::2]
        dataPos = dataTraPos + dataIndPos

        fileTraNeg = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana5289_neg.fasta'
        dataTraNeg = open(fileTraNeg, 'r').readlines()[1::2]
        fileIndNeg = '../Data/data_Lin2022Evaluation/A.thaliana/A.thaliana1000indep_neg.fasta'
        dataIndNeg = open(fileIndNeg, 'r').readlines()[1::2]
        dataNeg = dataTraNeg + dataIndNeg

        dataSeq = dataPos + dataNeg
        dataSeq = [sample.upper().strip() for sample in dataSeq]

        dataSeq = tRNA2DNA(dataSeq)
        dataY = np.array([1] * len(dataPos) + [0] * len(dataNeg))

        if False:
            sys.exit('Wrong!')
        elif False: # True:
            seqTra, seqVal, seqTes, yTra, yVal, yTes = self.splitTraValTes(dataSeq, dataY)
        elif True: # False:
            traRatio = 0.9
            # valRatio = 1 - traRatio

            seqTes = dataIndPos + dataIndNeg
            yTes = np.array([1] * len(dataIndPos) + [0] * len(dataIndNeg))
            traLenPos = int(len(dataTraPos) * traRatio)
            traLenNeg = int(len(dataTraNeg) * traRatio)

            dataTraPos_copy = copy.deepcopy(dataTraPos)
            dataTraNeg_copy = copy.deepcopy(dataTraNeg)
            random.seed(0)
            random.shuffle(dataTraPos_copy)
            random.shuffle(dataTraNeg_copy)

            seqTra = dataTraPos_copy[:traLenPos] + dataTraNeg_copy[:traLenNeg]
            yTra = np.array([1] * traLenPos + [0] * traLenNeg)
            seqVal = dataTraPos_copy[traLenPos:] + dataTraNeg_copy[traLenNeg:]
            yVal = np.array([1] * (len(dataTraPos_copy)-traLenPos) +
                            [0] * (len(dataTraNeg_copy)-traLenNeg))
        else:
            sys.exit('Wrong split type!')

        return dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes


    def getData_Liu2022Developmental(self, filePos, fileNeg):
        dataPos = open(filePos, 'r').readlines()[1::2]
        dataNeg = open(fileNeg, 'r').readlines()[1::2]

        dataSeq = dataPos + dataNeg
        dataSeq = [sample.upper().strip() for sample in dataSeq]

        dataY = np.array([1] * len(dataPos) + [0] * len(dataNeg))

        seqTra, seqVal, seqTes, yTra, yVal, yTes = self.splitTraValTes(dataSeq, dataY)

        return dataSeq, dataY, seqTra, seqVal, seqTes, yTra, yVal, yTes


    def getData_homo220123(self):
        dataPrefix = '../Data/homoData_220123/'
        namePos = os.path.join(dataPrefix, 'sequence_All_exon_all_Pos.csv')
        nameNeg = os.path.join(dataPrefix, r'sequence_All_exon_all_Neg.csv')

        def fRead(fileName):
            temp = [line.replace(',', '').lower().strip()
                    for line in open(fileName, 'r')]
            return temp
        dataPos = fRead(namePos)
        dataNeg = fRead(nameNeg)

        return dataPos, dataNeg


    def getData_Chen2020m5CPredSVM_A(self):
        prefix = '../Data/Chen2020m5CPredSVM_benchmark/'
        fileName_train_pos = os.path.join(prefix, 'A_train_pos.fasta')
        train_pos = open(fileName_train_pos, 'r').readlines()[1::2]

        fileName_train_neg = os.path.join(prefix, 'A_train_neg.fasta')
        train_neg = open(fileName_train_neg, 'r').readlines()[1::2]

        fileName_test_pos = os.path.join(prefix, 'A_test_pos.fasta')
        test_pos = open(fileName_test_pos, 'r').readlines()[1::2]

        fileName_test_neg = os.path.join(prefix, 'A_test_neg.fasta')
        test_neg = open(fileName_test_neg, 'r').readlines()[1::2]

        dataSeqPos = train_pos + test_pos
        dataSeqNeg = train_neg + test_neg

        dataSeq = dataSeqPos + dataSeqNeg
        dataSeq = [sample.upper().strip() for sample in dataSeq]
        dataSeq = tRNA2DNA(dataSeq)

        dataY = np.array([1] * len(dataSeqPos) + [0] * len(dataSeqNeg))

        return dataSeq, dataY


    def getData_Chen2020m5CPredSVM_H(self):
        prefix = '../Data/Chen2020m5CPredSVM_benchmark/'
        fileName_train_pos = os.path.join(prefix, 'H_train_pos.fasta')
        train_pos = open(fileName_train_pos, 'r').readlines()[1::2]

        fileName_train_neg = os.path.join(prefix, 'H_train_neg.fasta')
        train_neg = open(fileName_train_neg, 'r').readlines()[1::2]

        fileName_test_pos = os.path.join(prefix, 'H_test_pos.fasta')
        test_pos = open(fileName_test_pos, 'r').readlines()[1::2]

        fileName_test_neg = os.path.join(prefix, 'H_test_neg.fasta')
        test_neg = open(fileName_test_neg, 'r').readlines()[1::2]

        dataSeqPos = train_pos + test_pos
        dataSeqNeg = train_neg + test_neg

        dataSeq = dataSeqPos + dataSeqNeg
        dataSeq = [sample.upper().strip() for sample in dataSeq]
        dataSeq = tRNA2DNA(dataSeq)

        dataY = np.array([1] * len(dataSeqPos) + [0] * len(dataSeqNeg))

        return dataSeq, dataY

    def getData_Chen2020m5CPredSVM_M(self):
        prefix = '../Data/Chen2020m5CPredSVM_benchmark/'
        fileName_train_pos = os.path.join(prefix, 'M_train_pos.fasta')
        train_pos = open(fileName_train_pos, 'r').readlines()[1::2]

        fileName_train_neg = os.path.join(prefix, 'M_train_neg.fasta')
        train_neg = open(fileName_train_neg, 'r').readlines()[1::2]

        fileName_test_pos = os.path.join(prefix, 'M_test_pos.fasta')
        test_pos = open(fileName_test_pos, 'r').readlines()[1::2]

        fileName_test_neg = os.path.join(prefix, 'M_test_neg.fasta')
        test_neg = open(fileName_test_neg, 'r').readlines()[1::2]

        dataSeqPos = train_pos + test_pos
        dataSeqNeg = train_neg + test_neg

        dataSeq = dataSeqPos + dataSeqNeg
        dataSeq = [sample.upper().strip() for sample in dataSeq]
        dataSeq = tRNA2DNA(dataSeq)

        dataY = np.array([1] * len(dataSeqPos) + [0] * len(dataSeqNeg))

        return dataSeq, dataY

    def getData_Zhang2020iPromoter_5mC(self):
        prefix = '../Data/Zhang2020iPromoter_5mC/'
        fileName_pos = os.path.join(prefix, 'all_positive.fasta')
        dataSeqPos = open(fileName_pos, 'r').readlines()[1::2]

        fileName_neg = os.path.join(prefix, 'equal_negative.fasta')
        dataSeqNeg = open(fileName_neg, 'r').readlines()[1::2]

        dataSeq = dataSeqPos + dataSeqNeg
        dataSeq = [sample.upper().strip() for sample in dataSeq]

        dataY = np.array([1] * len(dataSeqPos) + [0] * len(dataSeqNeg))

        return dataSeq, dataY

# %% SplitData
class SplitData(object):
    # in: args['seed']
    # in: ratio-- train: validate: test
    def __init__(self, args, dataSeq, dataY):
        self.args = args
        dataDict = dict()

        seqTra, seqVal, seqTes, yTra, yVal, yTes = self.splitTraValTes(
            dataSeq, dataY)

        dataDict['seqTra'] = seqTra
        dataDict['seqVal'] = seqVal
        dataDict['seqTes'] = seqTes

        dataDict['yTra'] = yTra
        dataDict['yVal'] = yVal
        dataDict['yTes'] = yTes

        self.dataDict = dataDict
        return

    def splitTraValTes(self, dataSeq, dataY):
        # seed = 0
        # traRatio = 0.7
        # valRatio = 0.1
        # tesRatio = 0.2
        args = self.args
        seed = args['seed']
        traRatio = args['traRatio']
        valRatio = args['valRatio']
        tesRatio = args['tesRatio']

        indexPos = np.nonzero(dataY)[0]
        indexNeg = np.where(dataY == 0)[0]
        assert len(indexPos) + len(indexNeg) == len(dataY)

        random.seed(seed)
        random.shuffle(indexPos)
        random.seed(seed+1)
        random.shuffle(indexNeg)

        traNumPos = int(len(indexPos)*traRatio)
        traNumNeg = int(len(indexNeg)*traRatio)
        valNumPos = int(len(indexPos)*valRatio)
        valNumNeg = int(len(indexNeg)*valRatio)
        tesNumPos = int(len(indexPos)*tesRatio)
        tesNumNeg = int(len(indexNeg)*tesRatio)

        indexTra = np.hstack((indexPos[: traNumPos], indexNeg[: traNumNeg]))
        indexVal = np.hstack((indexPos[traNumPos: traNumPos + valNumPos],
                              indexNeg[traNumNeg: traNumNeg + valNumNeg]))
        indexTes = np.hstack((indexPos[traNumPos + valNumPos:],
                              indexNeg[traNumNeg + valNumNeg:]))

        seqTra = [dataSeq[ind] for ind in indexTra]
        seqVal = [dataSeq[ind] for ind in indexVal]
        seqTes = [dataSeq[ind] for ind in indexTes]

        yTra = dataY[np.array(indexTra)]
        yVal = dataY[np.array(indexVal)]
        yTes = dataY[np.array(indexTes)]

        return seqTra, seqVal, seqTes, yTra, yVal, yTes


def saveSequences(seqs, ys, filePrefix):
    pos_name = f'{filePrefix}_pos.fasta'
    neg_name = f'{filePrefix}_neg.fasta'
    if os.path.exists(pos_name):
        return
    pos_mask = ys == 1
    
    pos_seqs = []
    neg_seqs = []
    for i, seq in enumerate(seqs):
        if pos_mask[i]:
            pos_seqs.append(f'>{i}')
            pos_seqs.append(seq)
        else:
            neg_seqs.append(f'>{i}')
            neg_seqs.append(seq)
            
    with open(pos_name, 'w') as fobj:
        fobj.write('\n'.join(pos_seqs))
    with open(neg_name, 'w') as fobj:
        fobj.write('\n'.join(neg_seqs))
    
    return

# %% main
if __name__ == '__main__':
    argsObj = Args()
    args = argsObj.args

    dataObj = GetRawData(args)
    dataDict = dataObj.dataDict

    # dataSeq, dataY = dataObj.dataSeq, dataObj.dataY

    # splitObj = SplitData(args, dataSeq, dataY)
    # dataDict = splitObj.dataDict


# %% End
