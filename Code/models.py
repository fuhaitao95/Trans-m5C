# -*- coding: utf-8 -*-
"""
Created on Sun May 28 05:11:23 2023


"""


# %% Import
import copy
import itertools
import math
import numpy as np
import pandas as pd
import sys
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from repDNA.nac import RevcKmer
from repDNA.psenac import PseDNC
from repDNA.psenac import PCPseDNC
from repDNA.psenac import PCPseTNC
from repDNA.psenac import SCPseDNC
from repDNA.psenac import SCPseTNC

# from piRNAPredictor.GetVariousFeatures import GetKmerDict
# from piRNAPredictor.GetVariousFeatures import GetSpectrumProfile
# from piRNAPredictor.GetVariousFeatures import GetMismatchProfile

from utils.util import torchDataset

from utils.getNV import getFeatureNVSeqs




# %%% Seq2ind
class Seq2ind(object):
    def __init__(self, seqs, num_k):
        self.num_k = num_k

        kmer2ind, ind2kmer = self.getPermute()
        dataNum = self.str2num(seqs, kmer2ind)
        self.dataNum = dataNum

        return

    def getPermute(self):
        num_k = self.num_k
        f = ['A', 'C', 'G', 'T']
        c = itertools.product(*([f for i in range(num_k)]))

        ind2kmer = []
        kmer2ind = dict()
        ind2kmer.append('null')
        kmer2ind['null'] = 0
        for i, value in enumerate(c):
            temp = ''.join(value)
            ind2kmer.append(temp)
            kmer2ind[temp] = i+1

        return kmer2ind, ind2kmer

    def str2num(self, data, kmer2ind):
        num_k = self.num_k

        dataNum = []
        iSample = 0
        for iSample in range(len(data)):
            sample = data[iSample].strip().upper().replace('U', 'T')
            # assert len(sample) == length
            sampleNum = []
            for jSite in range(len(sample)-num_k+1):
                kMer_value = sample[jSite: jSite + num_k]
                if 'N' in kMer_value.upper():
                    sampleNum.append(0)
                else:
                    sampleNum.append(kmer2ind[kMer_value])
            dataNum.append(sampleNum)
        dataNum = np.array(dataNum)
        return dataNum


# %%% Train GRU
class TraGRU(object):
    def __init__(self, args, dataDict):
        self.args = args
        self.dataDict = dataDict

        self.initModel()

        return

    def initModel(self):
        args = self.args
        lr = args['lr']
        weight_decay = args['weight_decay']
        device = args['device']
        modelName = args['modelName']

        if False: 
            sys.exit('wrong!')
        elif modelName == 'Transformer':
            model = Transformer(args)
        else:
            sys.exit('wrong!')

        model = model.to(device)
        lossF = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)

        self.model = model
        self.lossF = lossF
        self.optimizer = optimizer
        return

    def train(self, args, XTra, yTra):
        epochs = bestEpoch = args['estEpoch']
        interval = args['interval']

        result = dict()

        epoch_ls = []
        Loss_tr_ls = []
        AUC_tra_ls = []
        AUPR_tr_ls = []

        numResult = math.floor(float(epochs)/interval) + 1
        if interval == 1:
            numResult = numResult - 1
        columnsTra = ['epoch', 'traLoss', 'traAUC', 'traAUPR']
        metricsTra = np.zeros((numResult, len(columnsTra)))

        try:
            for epoch in range(1, bestEpoch+1):
                print('--- Start training ---')
                epoch_start = time.time()

                for epoch in range(1, epochs+1):
                    loss_value, AUC_tra, AUPR_tr = self.trainEpoch(XTra, yTra)

                    if (epoch == 1) or (epoch % interval == 0):
                        print(f'---train info Epoch: {epoch}---\n'
                              f'train      AUC: {AUC_tra:.4f}\n')

                        epoch_ls.append(epoch)
                        Loss_tr_ls.append(loss_value)
                        AUC_tra_ls.append(AUC_tra)
                        AUPR_tr_ls.append(AUPR_tr)

                        time_use = round((time.time()-epoch_start)/epoch/60, 3)
                        print(f'The {epoch} epochs totally take {time_use} minutes')
        except KeyboardInterrupt:
            print(f'Epoch: {epoch}, KeyboardInterrupt by user')

        length = len(AUPR_tr_ls)
        metricsTra[:length, 0] = np.array(epoch_ls[:length])
        metricsTra[:length, 1] = np.array(Loss_tr_ls[:length])
        metricsTra[:length, 2] = np.array(AUC_tra_ls[:length])
        metricsTra[:length, 3] = np.array(AUPR_tr_ls[:length])
        metricsTra_pd = pd.DataFrame(metricsTra, columns=columnsTra)
        result['metricsTra_pd'] = metricsTra_pd

        return result

    def traValTes(self):
        dataDict = self.dataDict
        XTra, yTra = dataDict['seqTra'], dataDict['yTra']
        XVal, yVal = dataDict['seqVal'], dataDict['yVal']
        XTes, yTes = dataDict['seqTes'], dataDict['yTes']

        if not hasattr(self, 'model'):
            sys.exit('Please initial the model first.')

        args = self.args
        epochs = args['epochs']
        interval = args['interval']
        bestEpoch = args['bestEpoch']
        earlyFlag = args['earlyFlag']
        patience = args['patience']
        prefix = args['prefix']

        result = dict()

        auc_max = 0.
        count = 0
        epoch_ls = []

        Loss_tr_ls = []
        AUC_tra_ls = []
        AUPR_tr_ls = []

        AUC_val_ls = []
        AUPR_va_ls = []

        AUC_tes_ls = []
        AUPR_te_ls = []

        numResult = math.floor(float(epochs)/interval) + 1
        if interval == 1:
            numResult = numResult - 1
        columnsTra = ['epoch', 'traLoss', 'traAUC', 'traAUPR']
        metricsTra = np.zeros((numResult, len(columnsTra)))

        columnsVal = ['epoch', 'valLoss', 'valAUC', 'valAUPR']
        metricsVal = np.zeros((numResult, len(columnsVal)))

        columnsTes = ['epoch', 'tesLoss', 'tesLoss', 'tesAUC', 'tesAUPR']
        metricsTes = np.zeros((numResult, len(columnsTes)))

        try:
            print('--- Start training ---')
            epoch_start = time.time()

            for epoch in range(1, epochs+1):
                loss_value, AUC_tra, AUPR_tr = self.trainEpoch(args, XTra, yTra)

                count = count + 1
                if earlyFlag and (count > patience):
                    break

                AUC_val, AUPR_va = self.validate(XVal, yVal)

                AUC_tes, AUPR_te, predict_test = self.test(XTes, yTes)

                if AUC_val > auc_max:
                    auc_max = AUC_val
                    epoch_max = epoch
                    auc_tra_max = AUC_tra
                    AUPR_tr_max = AUPR_tr
                    auc_val_max = AUC_val
                    AUPR_va_max = AUPR_va
                    auc_tes_max = AUC_tes
                    AUPR_te_max = AUPR_te
                    predict_test_max = copy.deepcopy(predict_test)
                    print(f'---max info Epoch: {epoch_max}---\n'
                          f'train      AUC: {auc_tra_max:.4f}\n'
                          f'validation AUC: {auc_val_max:.4f}\n'
                          f'test       AUC: {auc_tes_max:.4f}\n')
                    count = 0

                if epoch == bestEpoch:
                    self.predict_test_best = copy.deepcopy(predict_test)
                if (epoch == 1) or (epoch % interval == 0):
                    print(f'---train info Epoch: {epoch}---\n'
                          f'train      AUC: {AUC_tra:.4f}\n'
                          f'validation AUC: {AUC_val:.4f}\n'
                          f'test       AUC: {AUC_tes:.4f}\n')

                    epoch_ls.append(epoch)

                    Loss_tr_ls.append(loss_value)
                    AUC_tra_ls.append(AUC_tra)
                    AUPR_tr_ls.append(AUPR_tr)

                    AUC_val_ls.append(AUC_val)
                    AUPR_va_ls.append(AUPR_va)

                    AUC_tes_ls.append(AUC_tes)
                    AUPR_te_ls.append(AUPR_te)

        except KeyboardInterrupt:
            print(f'Epoch: {epoch}, KeyboardInterrupt by user')

        print(f'---max info Epoch: {epoch_max}---\n'
              f'train      AUC: {auc_tra_max:.4f}\n'
              f'validation AUC: {auc_val_max:.4f}\n'
              f'test       AUC: {auc_tes_max:.4f}\n')

        time_use = round((time.time()-epoch_start)/60, 3)
        print(f'The {epoch} epochs totally take {time_use} minutes')

        result['auc_tra_max'], result['AUPR_tr_max'] = auc_tra_max, AUPR_tr_max
        result['auc_val_max'], result['AUPR_va_max'] = auc_val_max, AUPR_va_max
        result['auc_tes_max'], result['AUPR_te_max'] = auc_tes_max, AUPR_te_max

        length = len(AUPR_te_ls)
        metricsTra[:length, 0] = np.array(epoch_ls[:length])
        metricsTra[:length, 1] = np.array(Loss_tr_ls[:length])
        metricsTra[:length, 2] = np.array(AUC_tra_ls[:length])
        metricsTra[:length, 3] = np.array(AUPR_tr_ls[:length])
        metricsTra_pd = pd.DataFrame(metricsTra, columns=columnsTra)
        result['metricsTra_pd'] = metricsTra_pd

        metricsVal[:length, 0] = np.array(epoch_ls[:length])
        metricsVal[:length, 2] = np.array(AUC_val_ls[:length])
        metricsVal[:length, 3] = np.array(AUPR_va_ls[:length])
        metricsVal_pd = pd.DataFrame(metricsVal, columns=columnsVal)
        result['metricsVal_pd'] = metricsVal_pd

        metricsTes[:length, 0] = np.array(epoch_ls[:length])
        metricsTes[:length, 2] = np.array(AUC_tes_ls[:length])
        metricsTes[:length, 3] = np.array(AUPR_va_ls[:length])
        metricsTes_pd = pd.DataFrame(metricsTes, columns=columnsTes)
        result['metricsTes_pd'] = metricsTes_pd

        y_label_pred = np.vstack((yTes, predict_test_max)).T
        columns = ['tesLabel', 'tesPredictProb']
        y_label_pred_pd = pd.DataFrame(y_label_pred, columns=columns)
        result['y_label_pred_pd'] = y_label_pred_pd

        outName = (f'{prefix}_label_pred_max_epoch{epoch_max}_'
                   f'trainAUC{auc_tra_max:.4f}_'
                   f'validationAUC_{auc_val_max:.4f}_'
                   f'testAUC{auc_tes_max:.4f}.csv')
        result['y_label_pred_pd_name'] = outName

        return result

    def trainEpoch(self, args, XTra, yTra):
        model = self.model
        lossF = self.lossF
        optimizer = self.optimizer

        device = args['device']
        num_k = args['num_k']
        batchSize = args['batchSize']

        dataX = Seq2ind(XTra, num_k).dataNum
        traDataset = torchDataset(dataX, yTra)
        traLoader = DataLoader(dataset=traDataset,
                               batch_size=batchSize,
                               shuffle=True)
        lenTra = len(traDataset)

        y_ls = []
        pred_ls = []
        loss_total = 0
        model.train()
        for ind, (x, y) in enumerate(traLoader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            predict_value, otherDict = model(x)
            loss = lossF(predict_value, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            
            loss_total += loss.item() * len(y)
            
            temp_ar = predict_value.detach().cpu().numpy()
            if sum(np.isnan(temp_ar))>0:
                break
            
            y_ls += y.cpu().numpy().tolist()
            pred = torch.sigmoid(predict_value.detach())
            pred_ls += pred.cpu().numpy().tolist()

        loss_value = loss_total / lenTra
        
        if len(y_ls) == 0:
            AUC_tra = 0
            AUPR_tr = 0
        else:
            AUC_tra = round(roc_auc_score(y_ls, pred_ls), 4)
            
            AUPR_tr = round(average_precision_score(y_ls, pred_ls), 4)
        return loss_value, AUC_tra, AUPR_tr


    def predict(self, seqs):
        if not hasattr(self, 'model'):
            sys.exit('Please initial the model first.')
        model = self.model

        args = self.args
        num_k = args['num_k']
        device = args['device']

        dataX = Seq2ind(seqs, num_k).dataNum
        datasetX = torch.LongTensor(dataX)
        datasetX = datasetX.to(device)

        model.eval()
        with torch.no_grad():
            predict_value, otherDict = model(datasetX)
            pred = torch.sigmoid(predict_value).cpu().numpy()
        if sum(np.isnan(pred))>0:
            pred = np.array([0.5] * len(pred))
        return pred, otherDict

    def validate(self, XVal, yVal):
        predict_vali_proba, otherDict = self.predict(XVal)
        AUC_val = round(roc_auc_score(yVal, predict_vali_proba), 4)
        AUPR_va = round(average_precision_score(yVal, predict_vali_proba), 4)
        return AUC_val, AUPR_va

    def test(self, XTes, yTes):
        predict_test_proba, otherDict = self.predict(XTes)
        AUC_tes = round(roc_auc_score(yTes, predict_test_proba), 4)
        AUPR_te = round(average_precision_score(yTes, predict_test_proba), 4)
        return AUC_tes, AUPR_te, predict_test_proba

    def allPred(self, dataSeq, dataY):
        seq_num = int(len(dataY) / 10.)
        predict_test_proba_ls, otherDict_ls = [], []
        temp = seq_num
        for i in range(9):
            predict_test_proba_i, otherDict_i = self.predict(dataSeq[temp-seq_num:temp])
            temp = temp + seq_num
            predict_test_proba_ls.append(predict_test_proba_i)
            otherDict_ls.append(otherDict_i)
        predict_test_proba_i, otherDict_i = self.predict(dataSeq[temp-seq_num:])
        predict_test_proba_ls.append(predict_test_proba_i)
        otherDict_ls.append(otherDict_i)
        
        predict_test_proba = np.hstack(predict_test_proba_ls)
        enc_self_attns_all = []
        for layer_i in range(len(otherDict_ls[0]['enc_self_attns'])):
            enc_layer = [temp['enc_self_attns'][layer_i] for temp in otherDict_ls]
            enc_cat = np.concatenate(enc_layer, axis=0)
            enc_self_attns_all.append(enc_cat)
        
        otherDict = {'enc_self_attns': enc_self_attns_all}
        
        # predict_test_proba, otherDict = self.predict(dataSeq)
        AUC_All = round(roc_auc_score(dataY, predict_test_proba), 4)
        AUPR_Al = round(average_precision_score(dataY, predict_test_proba), 4)
        return AUC_All, AUPR_Al, predict_test_proba, otherDict
    

# %% Transformer

class NormRaw(nn.Module):
    def __init__(self):
        super(NormRaw, self).__init__()
        return
    def forward(self, x):
        return x


def getNorm(norm, nHid):
    if False:
        sys.exit('wrong!')
    elif norm == 'BatchNorm':
        normalization = nn.BatchNorm1d(nHid)
    elif norm == 'LayerNorm':
        normalization = nn.LayerNorm(nHid)
    elif norm == 'None':
        normalization = NormRaw()
    else:
        sys.exit('wrong normalization: {norm}')
    return normalization


class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()

        d_model = args['hidDim']
        # dropProb = args['dropProb']
        max_len = 1000 # args['sampleLen']

        # dropout = nn.Dropout(p=dropProb)
        # self.dropout = dropout

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: [batch_size, seq_len, d_model]
        self.register_buffer('pe', pe)
        return
    def forward(self, x):
        '''
        function of forward for positional encoding

        Parameters
        ----------
        x : torch.float
            shape: [batch_size, seq_len, d_model].

        Returns
        -------
        x.

        '''
        pe = self.pe
        # dropout = self.dropout

        seq_len = x.size(1)
        x = x + pe[:, :seq_len, :]
        # x = dropout(x)
        return x


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    for i in range(batch_size):
        query_zero = torch.where(seq_q[i]==0)[0]
        for index in query_zero:
            pad_attn_mask[i][index] = True
    return pad_attn_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        dropProb = args['dropProb']
        Dropout = nn.Dropout(dropProb)
        self.Dropout = Dropout
        return

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        Dropout = self.Dropout

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        attn = Dropout(attn)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        d_model = args['hidDim']
        self.d_model = d_model

        n_heads = args['n_heads']
        self.n_heads = n_heads

        assert d_model % n_heads == 0
        d_k = d_model // n_heads
        self.d_k = d_k

        # We assume d_v always equals d_k
        W_Q = nn.Linear(d_model, d_model)
        W_K = nn.Linear(d_model, d_model)
        W_V = nn.Linear(d_model, d_model)
        fc = nn.Linear(d_model, d_model)

        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.fc = fc

        attObj = ScaledDotProductAttention(args)
        self.attObj = attObj
        return
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        d_k = self.d_k
        n_heads = self.n_heads
        d_model = self.d_model

        W_Q = self.W_Q
        W_K = self.W_K
        W_V = self.W_V

        attObj = self.attObj

        fc = self.fc

        batch_size = input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # raw attn_mask: [batch_size, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = attObj(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, d_model) # context: [batch_size, len_q, d_model]
        output = fc(context) # [batch_size, len_q, d_model]
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        d_model = args['hidDim']
        d_ff = args['d_ff']


        self.fc = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff, d_model)
                                )
        return
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        output = self.fc(inputs)
        return output # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        d_model = args['hidDim']
        dropProb = args['dropProb']
        norm = args['norm']
        sampleLen = args['sampleLen']
        num_k = args['num_k']
        batch_norm_dim = sampleLen - num_k + 1

        enc_self_attn = MultiHeadAttention(args)
        self.enc_self_attn = enc_self_attn

        if norm == 'BatchNorm':
            attNorm = getNorm(norm, batch_norm_dim)
        else:
            attNorm = getNorm(norm, d_model)
        # attNorm = nn.LayerNorm(d_model)
        self.attNorm = attNorm

        attDrop = nn.Dropout(dropProb)
        self.attDrop = attDrop

        pos_ffn = PoswiseFeedForwardNet(args)
        self.pos_ffn = pos_ffn

        if norm == 'BatchNorm':
            ffnNorm = getNorm(norm, batch_norm_dim)
        else:
            ffnNorm = getNorm(norm, d_model) #nn.LayerNorm(d_model)
        self.ffnNorm = ffnNorm

        ffnDrop = nn.Dropout(dropProb)
        self.ffnDrop = ffnDrop
        return

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        enc_self_attn = self.enc_self_attn
        attNorm = self.attNorm
        attDrop = self.attDrop
        pos_ffn = self.pos_ffn
        ffnNorm = self.ffnNorm
        ffnDrop = self.ffnDrop

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        att_outputs, attn = enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # att_outputs to same Q,K,V
        enc_outputs = enc_inputs + att_outputs
        enc_outputs = attNorm(enc_outputs)
        enc_outputs = attDrop(enc_outputs)

        ffn_outputs = pos_ffn(enc_outputs) # ffn_outputs: [batch_size, src_len, d_model]
        ffn_outputs = enc_outputs + ffn_outputs
        ffn_outputs = ffnNorm(ffn_outputs)
        ffn_outputs = ffnDrop(ffn_outputs)
        return ffn_outputs, attn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        embeddingDim = args['embeddingDim']
        hidDim = args['hidDim']
        dropProb = args['dropProb']
        num_layers = args['num_layers']

        embed = self.getEmbed()
        self.embed = embed

        pos_emb = PositionalEncoding(args)
        self.pos_emb = pos_emb

        initDropout = nn.Dropout(dropProb)
        self.initDropout = initDropout

        linearProject = nn.Linear(embeddingDim, hidDim)
        self.linearProject = linearProject

        layers = nn.ModuleList([EncoderLayer(args) for _ in range(num_layers)])
        self.layers = layers
        return
    def getEmbed(self):
        args = self.args
        embedType = args['embedType']
        if False:
            sys.exit('wrong!')
        elif embedType.startswith('init'):
            # the init matrix borrow from paper -- Identifying enhancer–promoter
            # interactions with neural network based on pre-trained DNA
            embeddingFile = r'../Data/embedding_matrix.npy'
            embedding_init = np.load(embeddingFile)
            embedding_init = torch.FloatTensor(embedding_init)
            if False:
                sys.exit('Wrong!')
            elif embedType == 'init_dynamic':
                embed = nn.Embedding.from_pretrained(
                    embedding_init, freeze=False)
            elif embedType == 'init_freeze':
                embed = nn.Embedding.from_pretrained(
                    embedding_init, freeze=True)
            else:
                sys.exit(f'wrong embedType: {embedType}')
        else:
            sys.exit(f'wrong embedType: {embedType}')
        return embed

    def forward(self, enc_inputs):
        embed = self.embed
        linearProject = self.linearProject
        pos_emb = self.pos_emb
        initDropout = self.initDropout
        layers = self.layers

        # word embedding
        enc_outputs = embed(enc_inputs) # shape: [batch_size, seq_len, embeddingDim]
        # initial embedding projection
        enc_outputs = linearProject(enc_outputs) # shape: [batch_size, seq_len, hidDim]

        # Add positional encoding
        enc_outputs = pos_emb(enc_outputs) # shape: [batch_size, seq_len, hidDim]
        enc_outputs = initDropout(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn.detach().cpu().numpy())
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        num_linear = args['num_linear']
        hidDim = args['hidDim']
        outputSize = args['outputSize']

        seq_len = args['sampleLen'] - args['num_k'] + 1
        linSize = hidDim * seq_len

        lin = nn.ModuleList()
        for i in range(num_linear-1):
            linear_temp = nn.Linear(linSize, int(linSize/2))
            lin.append(linear_temp)

            act = nn.LeakyReLU()
            lin.append(act)

            linSize = int(linSize / 2)
        linear_temp = nn.Linear(linSize, outputSize)
        lin.append(linear_temp)
        self.lin = lin

        return
    def forward(self, enc_outputs):
        lin = self.lin

        batch_size = enc_outputs.shape[0]
        enc_outputs = enc_outputs.reshape(batch_size, -1)

        for i, layer in enumerate(lin):
            enc_outputs = layer(enc_outputs)

        # enc_outputs = torch.mean(enc_outputs, dim=1)
        output = torch.squeeze(enc_outputs)

        return output



class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        encoder = Encoder(args)
        self.encoder = encoder

        decoder = Decoder(args)
        self.decoder = decoder
        return
    def forward(self, X):
        encoder = self.encoder
        decoder = self.decoder

        enc_outputs, enc_self_attns = encoder(X)
        output = decoder(enc_outputs)

        otherDict = dict()
        otherDict['enc_self_attns'] = enc_self_attns
        return output, otherDict




# %% End
