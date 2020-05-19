import pandas as pd
import numpy as np 


def label_anomoly(labels):
    '''
    param:
        label: list of label strings
        
    return:
        list of label indexes which have anomoly 
    
    '''
    lis = []
    for i,row in enumerate(labels):
        if ':' in row:
            lis += [i]
    return lis


def feature_transform(features,dim):
    '''
    params:
    feature: list of strings;
    dim: dimension of feature space
    
    return: 
    a numpy matrix representing the feature space of data
    '''
    matrix = np.zeros((len(features), dim))
    #dict_vectors = {}
    for i,row in enumerate(features):
         # we reindex, so some rows have values 'nan'
        if pd.isnull(row):
            continue
        row_sp = row.split(' ')
        key_val = {int(entry.split(':')[0]):float(entry.split(':')[1]) for entry in row_sp}
        for key in key_val:
            matrix[i][key] = key_val[key]
        
        
    return matrix


def label_transform(labels):
    '''
    params:
    labels: list of strings;
    
    return: 
    dictionary where key: index of label value: list of data index which is classfied in this label
    '''
    label_dict = {}
    for i,row in enumerate(labels):
        
        # we reindex, so some rows have values 'nan'
        if  pd.isnull(row):
            continue
        row_sp = row.strip().split(',')
        key_val = {int(entry): i for entry in row_sp}
        
        for key in key_val:
            if key not in label_dict:
                label_dict[key] = [key_val[key]]
            else:
                label_dict[key] += [key_val[key]]
    
    return label_dict 