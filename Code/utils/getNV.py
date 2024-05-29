#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 23:46:51 2023

@author: fht
"""

import numpy as np
seq = ['A', 'A', 'C', 'T', 'A', 'T', 'C']
seq_str = ''.join(seq)

def getFeatureNV(seq_str):
    seq_str = seq_str.upper().replace('U', 'T')
    baseLs = list(set(seq_str))
    seq_n = len(seq_str)
    
    n_k_dict = {base: seq_str.count(base) for base in baseLs}
    
    base_total_distance = {base: 0 for base in baseLs}
    for i, base in enumerate(seq_str):
        # print(i, base)
        base_total_distance[base] += i
    assert sum(base_total_distance.values()) == seq_n * (0 + seq_n - 1) / 2
    
    miu_k = {key: base_total_distance[key] / n_k_dict[key] for key in n_k_dict.keys()}
    
    n_k_max = max(n_k_dict.values())
    
    D_j_k_dict = {base: [] for base in baseLs}
    for D_j in range(2, seq_n+1):
        D_j_k = {base: 0 for base in baseLs}
        for seq_i, base in enumerate(seq_str):
            if D_j <= n_k_dict[base]:
                part = (seq_i-miu_k[base])**D_j / (n_k_dict[base]**(D_j-1) * seq_n**(D_j-1))
                D_j_k[base] += part
        
        for base in D_j_k.keys():
            D_j_k_dict[base].append(D_j_k[base])
    feature = []
    for key in ['A', 'T', 'C', 'G']:
        if key not in D_j_k_dict.keys():
            temp = [0] * (seq_n+1 - 2)
        else:
            temp = D_j_k_dict[key]
        feature = feature + temp
    return feature

def getFeatureNVSeqs(seqs):
    featureLs = []
    for seq in seqs:
        feature_i = getFeatureNV(seq)
        featureLs.append(feature_i)
    featureAr = np.array(featureLs, dtype=float)
    return featureAr
