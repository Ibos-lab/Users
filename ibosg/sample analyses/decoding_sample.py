import numpy as np
import numpy.matlib as mt

from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import align_trials
from ephysvibe.task import task_constants

from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import os 
from matplotlib import cm
from matplotlib import pyplot as plt
import glob
import pickle

from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import permutations
seed = 2023



# def reorganize_data(data):
#     ps_a_sample=np.empty([num_iter, data[0]['perf sample']['sample'].shape[0]])*np.nan
#     ps_a_orient=np.empty([num_iter, data[0]['perf sample']['orientation'].shape[0]])*np.nan
#     ps_a_color=np.empty([num_iter, data[0]['perf sample']['color'].shape[0]])*np.nan
#     ps_a_neutral=np.empty([num_iter, data[0]['perf sample']['neutral'].shape[0]])*np.nan

#     pt_a_sample=np.empty([num_iter, data[0]['perf test']['sample'].shape[0]])*np.nan
#     pt_a_orient=np.empty([num_iter, data[0]['perf test']['orientation'].shape[0]])*np.nan
#     pt_a_color=np.empty([num_iter, data[0]['perf test']['color'].shape[0]])*np.nan
#     pt_a_neutral=np.empty([num_iter, data[0]['perf test']['neutral'].shape[0]])*np.nan

#     for i in range(num_iter):
#         ps_a_sample[i,:]   =   data[i]['perf sample']['sample']
#         ps_a_orient[i,:]   =   data[i]['perf sample']['orientation']
#         ps_a_color[i,:]    =   data[i]['perf sample']['color']
#         ps_a_neutral[i,:]  =   data[i]['perf sample']['neutral']
        
#         pt_a_sample[i,:]   =   data[i]['perf test']['sample']
#         pt_a_orient[i,:]   =   data[i]['perf test']['orientation']
#         pt_a_color[i,:]    =   data[i]['perf test']['color']
#         pt_a_neutral[i,:]  =   data[i]['perf test']['neutral']
#     ps_a=   {"sample":ps_a_sample, "orientation":ps_a_orient, "color":ps_a_color, "neutral":ps_a_neutral}
#     pt_a=   {"sample":pt_a_sample, "orientation":pt_a_orient, "color":pt_a_color, "neutral":pt_a_neutral}
#     p_a =   {"perf sample": ps_a, "perf test": pt_a}

#     return p_a
def reorganize_data(data):
    ps_a_sample     =   np.empty([num_iter, data[0]['sample'].shape[0], data[0]['sample'].shape[0]])*np.nan
    ps_a_orient     =   np.empty([num_iter, data[0]['orientation'].shape[0], data[0]['orientation'].shape[0]])*np.nan
    ps_a_color      =   np.empty([num_iter, data[0]['color'].shape[0], data[0]['color'].shape[0]])*np.nan
    ps_a_neutral    =   np.empty([num_iter, data[0]['neutral'].shape[0], data[0]['neutral'].shape[0]])*np.nan

    # pt_a_sample=np.empty([num_iter, data[0]['perf test']['sample'].shape[0]])*np.nan
    # pt_a_orient=np.empty([num_iter, data[0]['perf test']['orientation'].shape[0]])*np.nan
    # pt_a_color=np.empty([num_iter, data[0]['perf test']['color'].shape[0]])*np.nan
    # pt_a_neutral=np.empty([num_iter, data[0]['perf test']['neutral'].shape[0]])*np.nan


    # ps_z_sample=np.empty([num_iter, data[0]['zscored']['sample'].shape[0]])*np.nan
    # ps_z_orient=np.empty([num_iter, data[0]['zscored']['orientation'].shape[0]])*np.nan
    # ps_z_color=np.empty([num_iter, data[0]['zscored']['color'].shape[0]])*np.nan
    # ps_z_neutral=np.empty([num_iter, data[0]['zscored']['neutral'].shape[0]])*np.nan

    # pt_z_sample=np.empty([num_iter, data[0]['zscored']['perf test']['sample'].shape[0]])*np.nan
    # pt_z_orient=np.empty([num_iter, data[0]['zscored']['perf test']['orientation'].shape[0]])*np.nan
    # pt_z_color=np.empty([num_iter, data[0]['zscored']['perf test']['color'].shape[0]])*np.nan
    # pt_z_neutral=np.empty([num_iter, data[0]['zscored']['perf test']['neutral'].shape[0]])*np.nan

    for i in range(num_iter):
        ps_a_sample[i,:]   =   data[i]['sample']
        ps_a_orient[i,:]   =   data[i]['orientation']
        ps_a_color[i,:]    =   data[i]['color']
        ps_a_neutral[i,:]  =   data[i]['neutral']
        
        # pt_a_sample[i,:]   =   data[i]['perf test']['sample']
        # pt_a_orient[i,:]   =   data[i]['perf test']['orientation']
        # pt_a_color[i,:]    =   data[i]['perf test']['color']
        # pt_a_neutral[i,:]  =   data[i]['perf test']['neutral']
        
        # ps_z_sample[i,:]   =   data[i]['zscored']['sample']
        # ps_z_orient[i,:]   =   data[i]['zscored']['orientation']
        # ps_z_color[i,:]    =   data[i]['zscored']['color']
        # ps_z_neutral[i,:]  =   data[i]['zscored']['neutral']
        
        # pt_z_sample[i,:]   =   data[i]['zscored']['perf test']['sample']
        # pt_z_orient[i,:]   =   data[i]['zscored']['perf test']['orientation']
        # pt_z_color[i,:]    =   data[i]['zscored']['perf test']['color']
        # pt_z_neutral[i,:]  =   data[i]['zscored']['perf test']['neutral']


    ps_a=   {"sample":ps_a_sample, "orientation":ps_a_orient, "color":ps_a_color, "neutral":ps_a_neutral}
    # pt_a=   {"sample":pt_a_sample, "orientation":pt_a_orient, "color":pt_a_color, "neutral":pt_a_neutral}
    p_a =   {"perf sample": ps_a}# , "perf test": pt_a

    # ps_z=   {"sample":ps_z_sample, "orientation":ps_z_orient, "color":ps_z_color, "neutral":ps_z_neutral}
    # pt_z=   {"sample":pt_z_sample, "orientation":pt_z_orient, "color":pt_z_color, "neutral":pt_z_neutral}
    # p_z =   {"perf sample": ps_z, "perf test": pt_z}

    # p   =   {"averaged": p_a, "zscored": p_z}
    return p_a
    
def pick_train_test_trials(trials, train_ratio):
    tmp     =   np.random.permutation(trials)
    train   =   tmp[:int((trials.shape[0]*train_ratio))]
    test    =   tmp[int((trials.shape[0]*train_ratio)):]

    train_rep   =   mt.repmat(train, test.shape[0],1)
    test_rep    =   mt.repmat(test, train.shape[0],1)
    if np.where(test_rep-np.transpose(train_rep)==0)[0].shape[0]!=0:
        print("Warnings: similar trials for training and testing")

    return train, test

# def decoding_par(d):
    
#     dat=d['data']
#     num_cells   =   d['num selected lip']
#     num_train   =   30
#     num_test    =   10
#     test_train_ratio    =   1-num_test/num_train
    
#     sel_cells=np.random.permutation(len(dat))
#     sel_cells=sel_cells[:num_cells]
    
#     names       =   list(dat[i]['name'] for i in sel_cells)
#     data        =   list(dat[i]['Sample zscored'] for i in sel_cells)
#     datat       =   list(dat[i]['Test1 zscored'] for i in sel_cells)
#     sample_id   =   list(dat[i]['Sample Id'] for i in sel_cells) 
        
#     o1c1trials  =   list(np.where(sample_id[i]==11)[0] for i in range(num_cells))#
#     o1c5trials  =   list(np.where(sample_id[i]==15)[0] for i in range(num_cells))#range(num_cells)
#     o5c1trials  =   list(np.where(sample_id[i]==51)[0] for i in range(num_cells))#range(num_cells)
#     o5c5trials  =   list(np.where(sample_id[i]==55)[0] for i in range(num_cells))#range(num_cells)

#     o1trials  =   list(np.where(np.floor(sample_id[i]/10)==1)[0] for i in range(num_cells))#range(num_cells)
#     o5trials  =   list(np.where(np.floor(sample_id[i]/10)==5)[0] for i in range(num_cells))#range(num_cells)
#     c1trials  =   list(np.where(np.floor(sample_id[i]%10)==1)[0] for i in range(num_cells))#range(num_cells)
#     c5trials  =   list(np.where(np.floor(sample_id[i]%10)==5)[0] for i in range(num_cells))#range(num_cells)

#     ntrials  =   list(np.where(sample_id[i]==0)[0] for i in range(num_cells))#range(num_cells)
#     nntrials  =   list(np.where(sample_id[i]!=0)[0] for i in range(num_cells))#range(num_cells)

#     min_trial_s     =   list(np.min([o1c1trials[i].shape[0], o1c5trials[i].shape[0], o5c1trials[i].shape[0], o5c5trials[i].shape[0]]) for i in range(num_cells))
#     min_trial_s     =   list(np.min([o1trials[i].shape[0], o5trials[i].shape[0]]) for i in range(num_cells))
#     min_trial_c     =   list(np.min([c1trials[i].shape[0], c5trials[i].shape[0]]) for i in range(num_cells))
    
#     data_train_c    =   np.empty([data[0].shape[1], num_train*2, num_cells])
#     data_train_o    =   np.empty([data[0].shape[1], num_train*2, num_cells])
#     data_train_n    =   np.empty([data[0].shape[1], num_train*2, num_cells])
#     data_train_s    =   np.empty([data[0].shape[1], num_train*4, num_cells])
    
#     datat_train_c   =   np.empty([datat[0].shape[1], num_train*2, num_cells])
#     datat_train_o   =   np.empty([datat[0].shape[1], num_train*2, num_cells])
#     datat_train_n   =   np.empty([datat[0].shape[1], num_train*2, num_cells])
#     datat_train_s   =   np.empty([datat[0].shape[1], num_train*4, num_cells])
    
#     data_test_c     = np.empty([data[0].shape[1], num_test*2, num_cells])
#     data_test_o     = np.empty([data[0].shape[1], num_test*2, num_cells])
#     data_test_n     = np.empty([data[0].shape[1], num_test*2, num_cells])
#     data_test_s     = np.empty([data[0].shape[1], num_test*4, num_cells])
    
#     datat_test_c    = np.empty([datat[0].shape[1], num_test*2, num_cells])
#     datat_test_o    = np.empty([datat[0].shape[1], num_test*2, num_cells])
#     datat_test_n    = np.empty([datat[0].shape[1], num_test*2, num_cells])
#     datat_test_s    = np.empty([datat[0].shape[1], num_test*4, num_cells])

#     y_train     =   np.concatenate([np.zeros(num_train), np.ones(num_train)])
#     y_test      =   np.concatenate([np.zeros(num_test), np.ones(num_test)])
#     y_train_s   =   np.concatenate([np.zeros(num_train), np.ones(num_train), np.zeros(num_train)+2, np.ones(num_train)+2])
#     y_test_s    =   np.concatenate([np.zeros(num_test), np.ones(num_test), np.zeros(num_test)+2, np.ones(num_test)+2])

#     perf_c      =    np.empty([len(data_train_c)])
#     perf_o      =    np.empty([len(data_train_o)])
#     perf_n      =    np.empty([len(data_train_n)])
#     perf_s      =    np.empty([len(data_train_s)])
    
#     perft_c     =    np.empty([len(datat_train_c)])
#     perft_o     =    np.empty([len(datat_train_o)])
#     perft_n     =    np.empty([len(datat_train_o)])
#     perft_s     =    np.empty([len(datat_train_s)])
    
    
#     trials_train_c  =   []
#     trials_train_o  =   []
#     trials_train_n  =   []
#     trials_train_s  =   []

#     trials_test_c   =   []
#     trials_test_o   =   []
#     trials_test_n   =   []
#     trials_test_s   =   []

#     for i in range(num_cells):
#         # color trials
#         train_c1, test_c1    =  pick_train_test_trials(c1trials[i], test_train_ratio)
#         train_c5, test_c5    =  pick_train_test_trials(c5trials[i], test_train_ratio)
#         trials_train_c.append(np.concatenate([np.random.choice(train_c1, num_train), np.random.choice(train_c5, num_train)]))
#         trials_test_c.append(np.concatenate([np.random.choice(test_c1, num_test), np.random.choice(test_c5, num_test)]))
        
#         # orientation trials  
#         train_o1, test_o1    =  pick_train_test_trials(o1trials[i], test_train_ratio)
#         train_o5, test_o5    =  pick_train_test_trials(o5trials[i], test_train_ratio)
#         trials_train_o.append(np.concatenate([np.random.choice(train_o1, num_train), np.random.choice(train_o5, num_train)]))
#         trials_test_o.append(np.concatenate([np.random.choice(test_o1, num_test), np.random.choice(test_o5, num_test)]))
                
#         # neutral trials  
#         train_n, test_n     =  pick_train_test_trials(ntrials[i], test_train_ratio)
#         train_nn, test_nn   =  pick_train_test_trials(nntrials[i], test_train_ratio)
#         trials_train_n.append(np.concatenate([np.random.choice(train_n, num_train), np.random.choice(train_nn, num_train)]))
#         trials_test_n.append(np.concatenate([np.random.choice(test_n, num_test), np.random.choice(test_nn, num_test)]))
                
#         # sample trials
#         train_o1c1, test_o1c1    =  pick_train_test_trials(o1c1trials[i], test_train_ratio)
#         train_o1c5, test_o1c5    =  pick_train_test_trials(o1c5trials[i], test_train_ratio)
#         train_o5c1, test_o5c1    =  pick_train_test_trials(o5c1trials[i], test_train_ratio)
#         train_o5c5, test_o5c5    =  pick_train_test_trials(o5c5trials[i], test_train_ratio)
#         trials_train_s.append(np.concatenate([np.random.choice(train_o1c1, num_train), np.random.choice(train_o1c5, num_train), np.random.choice(train_o5c1, num_train), np.random.choice(train_o5c5, num_train)]))
#         trials_test_s.append(np.concatenate([np.random.choice(test_o1c1, num_test), np.random.choice(test_o1c5, num_test), np.random.choice(test_o5c1, num_test), np.random.choice(test_o5c5, num_test)]))

#         # build matrices of  [timestamp, trials, neurons] dimensions to feed to classifiers
#         data_train_c[:, :, i]    = np.transpose(data[i][trials_train_c[i]])
#         data_train_o[:, :, i]    = np.transpose(data[i][trials_train_o[i]])
#         data_train_n[:, :, i]    = np.transpose(data[i][trials_train_n[i]])
#         data_train_s[:, :, i]    = np.transpose(data[i][trials_train_s[i]])

#         data_test_c[:, :, i]    = np.transpose(data[i][trials_test_c[i]])
#         data_test_o[:, :, i]    = np.transpose(data[i][trials_test_o[i]])
#         data_test_n[:, :, i]    = np.transpose(data[i][trials_test_n[i]])
#         data_test_s[:, :, i]    = np.transpose(data[i][trials_test_s[i]])
        
#         datat_train_c[:, :, i]   = np.transpose(datat[i][trials_train_c[i]])
#         datat_train_o[:, :, i]   = np.transpose(datat[i][trials_train_o[i]])
#         datat_train_n[:, :, i]   = np.transpose(datat[i][trials_train_n[i]])
#         datat_train_s[:, :, i]   = np.transpose(datat[i][trials_train_s[i]])

#         datat_test_c[:, :, i]   = np.transpose(datat[i][trials_test_c[i]])
#         datat_test_o[:, :, i]   = np.transpose(datat[i][trials_test_o[i]])
#         datat_test_n[:, :, i]   = np.transpose(datat[i][trials_test_n[i]])
#         datat_test_s[:, :, i]   = np.transpose(datat[i][trials_test_s[i]])

#     for t in range(len(data_train_c)):
#         model.fit(data_train_c[t],y_train)
#         y_predict = model.predict(data_test_c[t])
#         perf_c[t]=np.where(y_predict-y_test==0)[0].shape[0]
        
#         model.fit(data_train_o[t],y_train)
#         y_predict = model.predict(data_test_o[t])
#         perf_o[t]=np.where(y_predict-y_test==0)[0].shape[0]

#         model.fit(data_train_n[t],y_train)
#         y_predict = model.predict(data_test_n[t])
#         perf_n[t]=np.where(y_predict-y_test==0)[0].shape[0]

#         model.fit(data_train_s[t],y_train_s)
#         y_predict = model.predict(data_test_s[t])
#         perf_s[t]=np.where(y_predict-y_test_s==0)[0].shape[0]

#         if t< datat_train_c.shape[0]:
#             model.fit(datat_train_c[t],y_train)
#             y_predict = model.predict(datat_test_c[t])
#             perft_c[t]=np.where(y_predict-y_test==0)[0].shape[0]
            
#             model.fit(datat_train_o[t],y_train)
#             y_predict = model.predict(datat_test_o[t])
#             perft_o[t]=np.where(y_predict-y_test==0)[0].shape[0]

#             model.fit(datat_train_n[t],y_train)
#             y_predict = model.predict(datat_test_n[t])
#             perft_n[t]=np.where(y_predict-y_test==0)[0].shape[0]

#             model.fit(datat_train_s[t],y_train_s)
#             y_predict = model.predict(datat_test_s[t])
#             perft_s[t]=np.where(y_predict-y_test_s==0)[0].shape[0]

#     perfa_sample=   {"sample": perf_s, "orientation": perf_o, "color":perf_c, "neutral": perf_n}
#     perfa_test  =   {"sample": perft_s, "orientation": perft_o, "color":perft_c, "neutral": perft_n}
#     perf        =   {"perf sample": perfa_sample, "perf test": perfa_test}

#     return perf

def decoding_par(d):
# for ii in range(1):
#     d=included_lip
    
    dat=d['data']
    num_cells   =   d['num selected lip']
    # num_iter    =   1000
    num_train   =   30
    num_test    =   10
    test_train_ratio    =   1-num_test/num_train
    
    sel_cells=np.random.permutation(len(dat))
    sel_cells=sel_cells[:num_cells]
    
    names       =   list(dat[i]['name'] for i in sel_cells)
    datas       =   list(dat[i]['Sample zscored'] for i in sel_cells)
    datat       =   list(dat[i]['Test1 zscored'] for i in sel_cells)
    sample_id   =   list(dat[i]['Sample Id'] for i in sel_cells) 
     
    o1c1trials  =   list(np.where(sample_id[i]==11)[0] for i in range(num_cells))#
    o1c5trials  =   list(np.where(sample_id[i]==15)[0] for i in range(num_cells))#range(num_cells)
    o5c1trials  =   list(np.where(sample_id[i]==51)[0] for i in range(num_cells))#range(num_cells)
    o5c5trials  =   list(np.where(sample_id[i]==55)[0] for i in range(num_cells))#range(num_cells)

    o1trials    =   list(np.where(np.floor(sample_id[i]/10)==1)[0] for i in range(num_cells))#range(num_cells)
    o5trials    =   list(np.where(np.floor(sample_id[i]/10)==5)[0] for i in range(num_cells))#range(num_cells)
    c1trials    =   list(np.where(np.floor(sample_id[i]%10)==1)[0] for i in range(num_cells))#range(num_cells)
    c5trials    =   list(np.where(np.floor(sample_id[i]%10)==5)[0] for i in range(num_cells))#range(num_cells)

    ntrials     =   list(np.where(sample_id[i]==0)[0] for i in range(num_cells))#range(num_cells)
    nntrials    =   list(np.where(sample_id[i]!=0)[0] for i in range(num_cells))#range(num_cells)

    data_train_c    =   np.empty([datas[0].shape[1]+datat[0].shape[1], num_train*2, num_cells])
    data_train_o    =   np.empty([datas[0].shape[1]+datat[0].shape[1], num_train*2, num_cells])
    data_train_n    =   np.empty([datas[0].shape[1]+datat[0].shape[1], num_train*2, num_cells])
    data_train_s    =   np.empty([datas[0].shape[1]+datat[0].shape[1], num_train*4, num_cells])
    
    # datat_train_c   =   np.empty([datat[0].shape[1], num_train*2, num_cells])
    # datat_train_o   =   np.empty([datat[0].shape[1], num_train*2, num_cells])
    # datat_train_n   =   np.empty([datat[0].shape[1], num_train*2, num_cells])
    # datat_train_s   =   np.empty([datat[0].shape[1], num_train*4, num_cells])
    
    data_test_c     = np.empty([datas[0].shape[1]+datat[0].shape[1], num_test*2, num_cells])
    data_test_o     = np.empty([datas[0].shape[1]+datat[0].shape[1], num_test*2, num_cells])
    data_test_n     = np.empty([datas[0].shape[1]+datat[0].shape[1], num_test*2, num_cells])
    data_test_s     = np.empty([datas[0].shape[1]+datat[0].shape[1], num_test*4, num_cells])
    
    # datat_test_c    = np.empty([datat[0].shape[1], num_test*2, num_cells])
    # datat_test_o    = np.empty([datat[0].shape[1], num_test*2, num_cells])
    # datat_test_n    = np.empty([datat[0].shape[1], num_test*2, num_cells])
    # datat_test_s    = np.empty([datat[0].shape[1], num_test*4, num_cells])

    y_train     =   np.concatenate([np.zeros(num_train), np.ones(num_train)])
    y_test      =   np.concatenate([np.zeros(num_test), np.ones(num_test)])
    y_train_s   =   np.concatenate([np.zeros(num_train), np.ones(num_train), np.zeros(num_train)+2, np.ones(num_train)+2])
    y_test_s    =   np.concatenate([np.zeros(num_test), np.ones(num_test), np.zeros(num_test)+2, np.ones(num_test)+2])

    perf_c      =    np.empty([len(data_train_c), len(data_train_c)])
    perf_o      =    np.empty([len(data_train_o), len(data_train_o)])
    perf_n      =    np.empty([len(data_train_n), len(data_train_n)])
    perf_s      =    np.empty([len(data_train_s), len(data_train_s)])
    
    # perft_c     =    np.empty([len(datat_train_c), len(datat_train_c)])
    # perft_o     =    np.empty([len(datat_train_o), len(datat_train_o)])
    # perft_n     =    np.empty([len(datat_train_o), len(datat_train_n)])
    # perft_s     =    np.empty([len(datat_train_s), len(datat_train_s)])


    # perf_ct      =    np.empty([len(data_train_c), len(datat_train_c)])
    # perf_ot      =    np.empty([len(data_train_o), len(datat_train_o)])
    # perf_nt      =    np.empty([len(data_train_n), len(datat_train_n)])
    # perf_st      =    np.empty([len(data_train_s), len(datat_train_s)])
    
    # perft_cs     =    np.empty([len(datat_train_c), len(data_train_c)])
    # perft_os     =    np.empty([len(datat_train_o), len(data_train_o)])
    # perft_ns     =    np.empty([len(datat_train_o), len(data_train_n)])
    # perft_ss     =    np.empty([len(datat_train_s), len(data_train_s)])
    

    for cell in range(num_cells):
        # color trials
        train_c1, test_c1    =  pick_train_test_trials(c1trials[cell], test_train_ratio)
        train_c5, test_c5    =  pick_train_test_trials(c5trials[cell], test_train_ratio)
        trials_train_c  =   np.concatenate([np.random.choice(train_c1, num_train), np.random.choice(train_c5, num_train)])
        trials_test_c   =   np.concatenate([np.random.choice(test_c1, num_test), np.random.choice(test_c5, num_test)])
        
        # orientation trials  
        train_o1, test_o1    =  pick_train_test_trials(o1trials[cell], test_train_ratio)
        train_o5, test_o5    =  pick_train_test_trials(o5trials[cell], test_train_ratio)
        trials_train_o  =   np.concatenate([np.random.choice(train_o1, num_train), np.random.choice(train_o5, num_train)])
        trials_test_o   =   np.concatenate([np.random.choice(test_o1, num_test), np.random.choice(test_o5, num_test)])
                
        # neutral trials  
        train_n, test_n     =  pick_train_test_trials(ntrials[cell], test_train_ratio)
        train_nn, test_nn   =  pick_train_test_trials(nntrials[cell], test_train_ratio)
        trials_train_n  =   np.concatenate([np.random.choice(train_n, num_train), np.random.choice(train_nn, num_train)])
        trials_test_n   =   np.concatenate([np.random.choice(test_n, num_test), np.random.choice(test_nn, num_test)])
                
        # sample trials
        train_o1c1, test_o1c1    =  pick_train_test_trials(o1c1trials[cell], test_train_ratio)
        train_o1c5, test_o1c5    =  pick_train_test_trials(o1c5trials[cell], test_train_ratio)
        train_o5c1, test_o5c1    =  pick_train_test_trials(o5c1trials[cell], test_train_ratio)
        train_o5c5, test_o5c5    =  pick_train_test_trials(o5c5trials[cell], test_train_ratio)
        trials_train_s  =   np.concatenate([np.random.choice(train_o1c1, num_train), np.random.choice(train_o1c5, num_train), np.random.choice(train_o5c1, num_train), np.random.choice(train_o5c5, num_train)])
        trials_test_s   =   np.concatenate([np.random.choice(test_o1c1, num_test), np.random.choice(test_o1c5, num_test), np.random.choice(test_o5c1, num_test), np.random.choice(test_o5c5, num_test)])

        # build matrices of  [timestamp, trials, neurons] dimensions to feed to classifiers
        
        data_train_c[:, :, cell]    =   np.concatenate([np.transpose(datas[cell][trials_train_c]), np.transpose(datat[cell][trials_train_c])])
        data_train_o[:, :, cell]    =   np.concatenate([np.transpose(datas[cell][trials_train_o]), np.transpose(datat[cell][trials_train_o])])
        data_train_n[:, :, cell]    =   np.concatenate([np.transpose(datas[cell][trials_train_n]), np.transpose(datat[cell][trials_train_n])])
        data_train_s[:, :, cell]    =   np.concatenate([np.transpose(datas[cell][trials_train_s]), np.transpose(datat[cell][trials_train_s])])
        
        data_test_c[:, :, cell]     =   np.concatenate([np.transpose(datas[cell][trials_test_c]), np.transpose(datat[cell][trials_test_c])])
        data_test_o[:, :, cell]     =   np.concatenate([np.transpose(datas[cell][trials_test_o]), np.transpose(datat[cell][trials_test_o])])
        data_test_n[:, :, cell]     =   np.concatenate([np.transpose(datas[cell][trials_test_n]), np.transpose(datat[cell][trials_test_n])])
        data_test_s[:, :, cell]     =   np.concatenate([np.transpose(datas[cell][trials_test_s]), np.transpose(datat[cell][trials_test_s])])
        
        # data_train_c[:datas[i].shape[1], :, i]   = np.transpose(datas[i][trials_train_c[i]])
        # data_train_o[:datas[i].shape[1], :, i]   = np.transpose(datas[i][trials_train_o[i]])
        # data_train_n[:datas[i].shape[1], :, i]   = np.transpose(datas[i][trials_train_n[i]])
        # data_train_s[:datas[i].shape[1], :, i]   = np.transpose(datas[i][trials_train_s[i]])

        # data_train_c[datas[i].shape[1]:, :, i]   = np.transpose(datat[i][trials_train_c[i]])
        # data_train_o[datas[i].shape[1]:, :, i]   = np.transpose(datat[i][trials_train_o[i]])
        # data_train_n[datas[i].shape[1]:, :, i]   = np.transpose(datat[i][trials_train_n[i]])
        # data_train_s[datas[i].shape[1]:, :, i]   = np.transpose(datat[i][trials_train_s[i]])

        # data_test_c[datas[i].shape[1]:, :, i]    = np.transpose(datat[i][trials_test_c[i]])
        # data_test_o[datas[i].shape[1]:, :, i]    = np.transpose(datat[i][trials_test_o[i]])
        # data_test_n[datas[i].shape[1]:, :, i]    = np.transpose(datat[i][trials_test_n[i]])
        # data_test_s[datas[i].shape[1]:, :, i]    = np.transpose(datat[i][trials_test_s[i]])        
        
        # data_test_c[:datas[i].shape[1], :, i]    = np.transpose(datas[i][trials_test_c[i]])
        # data_test_o[:datas[i].shape[1], :, i]    = np.transpose(datas[i][trials_test_o[i]])
        # data_test_n[:datas[i].shape[1], :, i]    = np.transpose(datas[i][trials_test_n[i]])
        # data_test_s[:datas[i].shape[1], :, i]    = np.transpose(datas[i][trials_test_s[i]])
        
        # datat_train_c[:, :, i]   = np.transpose(datat[i][trials_train_c[i]])
        # datat_train_o[:, :, i]   = np.transpose(datat[i][trials_train_o[i]])
        # datat_train_n[:, :, i]   = np.transpose(datat[i][trials_train_n[i]])
        # datat_train_s[:, :, i]   = np.transpose(datat[i][trials_train_s[i]])

        # datat_test_c[:, :, i]   = np.transpose(datat[i][trials_test_c[i]])
        # datat_test_o[:, :, i]   = np.transpose(datat[i][trials_test_o[i]])
        # datat_test_n[:, :, i]   = np.transpose(datat[i][trials_test_n[i]])
        # datat_test_s[:, :, i]   = np.transpose(datat[i][trials_test_s[i]])

        # datas_train_c[:datas_train_c.shape[0],:,i]  =   datas_train_c[:, :, i]

    for time_train in range(data_train_c.shape[0]):
        
        
        model.fit(data_train_c[time_train],y_train)
        for time_test in range(data_train_c.shape[0]):
            y_predict = model.predict(data_test_c[time_test])
            perf_c[time_train,time_test]=np.where(y_predict-y_test==0)[0].shape[0]
            

        model.fit(data_train_o[time_train],y_train)
        for time_test in range(data_train_o.shape[0]):
            y_predict = model.predict(data_test_o[time_test])
            perf_o[time_train,time_test]=np.where(y_predict-y_test==0)[0].shape[0]
            
        model.fit(data_train_n[time_train],y_train)
        for time_test in range(data_train_n.shape[0]):
            y_predict = model.predict(data_test_n[time_test])
            perf_n[time_train,time_test]=np.where(y_predict-y_test==0)[0].shape[0]

        model.fit(data_train_s[time_train],y_train_s)
        for time_test in range(data_train_s.shape[0]):
            y_predict = model.predict(data_test_s[time_test])
            perf_s[time_train,time_test]=np.where(y_predict-y_test_s==0)[0].shape[0]
        

        # # model.fit(data_train_o[t],y_train)
        # for l in range(data_train_o.shape[1]):
        #     y_predict = model.predict(data_test_o[t])
        #     perf_o[t,l]=np.where(y_predict-y_test==0)[0].shape[0]

        
        # for l in range(data_train_n.shape[1]):
        #     y_predict = model.predict(data_test_o[t])
        #     perf_n[t,l]=np.where(y_predict-y_test==0)[0].shape[0]

        # model.fit(data_train_s[t],y_train_s)
        # y_predict = model.predict(data_test_s[t])
        # perf_s[t,l]=np.where(y_predict-y_test_s==0)[0].shape[0]

        # if t< datat_train_c.shape[0]:
        #     model.fit(datat_train_c[t],y_train)
        #     for l in range()
        #     y_predict = model.predict(datat_test_c[t])
        #     perft_c[t]=np.where(y_predict-y_test==0)[0].shape[0]
            
        #     model.fit(datat_train_o[t],y_train)
        #     y_predict = model.predict(datat_test_o[t])
        #     perft_o[t]=np.where(y_predict-y_test==0)[0].shape[0]

        #     model.fit(datat_train_n[t],y_train)
        #     y_predict = model.predict(datat_test_n[t])
        #     perft_n[t]=np.where(y_predict-y_test==0)[0].shape[0]

        #     model.fit(datat_train_s[t],y_train_s)
        #     y_predict = model.predict(datat_test_s[t])
        #     perft_s[t]=np.where(y_predict-y_test_s==0)[0].shape[0]


    perf=   {"sample": perf_s, "orientation": perf_o, "color":perf_c, "neutral": perf_n}
    # perfa_test  =   {"sample": perft_s, "orientation": perft_o, "color":perft_c, "neutral": perft_n}
    # perf        =   {"perf sample": perfa_sample, "perf test": perfa_test}

    # perfz_sample=   {"sample": perfz_s, "orientation": perfz_o, "color":perfz_c, "neutral":perfz_n}
    # perfz_test  =   {"sample": perfzt_s, "orientation": perfzt_o, "color":perfzt_c, "neutral":perfzt_n}
    # perf_zsc    =   {"perf sample": perfz_sample, "perf test": perfz_test}
    
    # perf        =   {"averaged": perf_avg, "zscored": perf_zsc}
    return perf



model=  SVC(kernel='linear',C=0.8 ,decision_function_shape='ovr',gamma='auto',degree=1)

do_lip  =   1
do_pfc  =   1
do_v4   =   1    
num_iter    =   1000

with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/LIPavacti', 'rb') as handle:
    lip = pickle.load(handle)


## decode information in LIP
# select trials for LIP
# av_lip=np.empty([len(lip), lip[0][0]['Sample averaged'].shape[1]])
# max_lip=[]
# mean_lip=[]
tmplip=[]
for i in range(len(lip)):
    # av_lip[i,:]=np.mean(lip[0][i]['Sample averaged'], axis=0)*1000
    # max_lip.append(np.max(av_lip[i,:]))
    # mean_lip.append(np.mean(av_lip[i,:]))
    # if np.max(av_lip[i,:])>=0:
    min_sample=np.min([np.where(lip[0][i]['Sample Id']==11)[0].shape, np.where(lip[0][i]['Sample Id']==15)[0].shape, np.where(lip[0][i]['Sample Id']==51)[0].shape, np.where(lip[0][i]['Sample Id']==55)[0].shape])
    if min_sample>=25:
        tmplip.append(lip[0][i])


included_lip={"data": tmplip, "num selected lip": len(tmplip)}
if do_lip==1:
    
    perf_lip    =   Parallel(n_jobs = -1)(delayed(decoding_par)(included_lip) for cell in tqdm(range(num_iter)))

    # with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/decoding_perf_lip_raw', "wb") as fp: 
    #     pickle.dump(perf_lip, fp)

    p_lip   =   reorganize_data(perf_lip)
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/decoding_perf_lipC0.8', "wb") as fp: 
        pickle.dump(p_lip, fp)

if do_pfc==1:
    
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/pfcavacti', 'rb') as handle:
        pfc = pickle.load(handle)

    ## decode information in PFC
    # select trials for PFC
    # av_pfc=np.empty([len(pfc), pfc[0][0]['Sample averaged'].shape[1]])
    # max_pfc=[]
    # mean_pfc=[]
    tmppfc=[]
    for i in range(len(pfc)):
        # av_pfc[i,:]=np.mean(pfc[0][i]['Sample averaged'], axis=0)*1000
        # max_pfc.append(np.max(av_pfc[i,:]))
        # mean_pfc.append(np.mean(av_pfc[i,:]))
        # if np.mean(av_pfc[i,:])>=5:
        #     tmppfc.append(pfc[i])
        min_sample=np.min([np.where(pfc[0][i]['Sample Id']==11)[0].shape, np.where(pfc[0][i]['Sample Id']==15)[0].shape, np.where(pfc[0][i]['Sample Id']==51)[0].shape, np.where(pfc[0][i]['Sample Id']==55)[0].shape])
        if min_sample>=25:
            tmppfc.append(pfc[0][i])

    included_pfc={"data": tmppfc, "num selected lip": len(tmplip)}

    perf_pfc    =   Parallel(n_jobs = 24)(delayed(decoding_par)(included_pfc) for cell in tqdm(range(num_iter)))
    # perf_pfc=[]
    # for l in tqdm(range(num_iter)):
    #     tmp=decoding_par(included_pfc)
    #     perf_pfc.append(tmp)
    # with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/decoding_perf_pfc_raw', "wb") as fp: 
    #     pickle.dump(perf_pfc, fp)
        
    p_pfc   =   reorganize_data(perf_pfc)
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/decoding_perf_pfcC0.8', "wb") as fp: 
        pickle.dump(p_pfc, fp)
if do_v4==1:
    
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/v4avacti', 'rb') as handle:
        v4 = pickle.load(handle)    
    ## decode information in V4
    # select trials for V4
    av_v4=np.empty([len(v4), v4[0][0]['Sample averaged'].shape[1]])
    max_v4=[]
    mean_v4=[]
    tmpv4=[]
    for i in range(len(v4)):
        av_v4[i,:]=np.mean(v4[0][i]['Sample averaged'], axis=0)*1000
        max_v4.append(np.max(av_v4[i,:]))
        mean_v4.append(np.mean(av_v4[i,:]))
        # if np.mean(av_v4[i,:])>=5:
        #     tmpv4.append(v4[i])
        min_sample=np.min([np.where(v4[0][i]['Sample Id']==11)[0].shape, np.where(v4[0][i]['Sample Id']==15)[0].shape, np.where(v4[0][i]['Sample Id']==51)[0].shape, np.where(v4[0][i]['Sample Id']==55)[0].shape])
        if min_sample>=25:
            tmpv4.append(v4[0][i])

    included_v4={"data": tmpv4, "num selected lip": len(tmplip)}
    # perf_v4=[]
    # for l in tqdm(range(num_iter)):
    #     tmp=decoding_par(included_v4)
    #     perf_v4.append(tmp)
    perf_v4 =   Parallel(n_jobs = 24)(delayed(decoding_par)(included_v4) for cell in tqdm(range(num_iter)))

    # with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/decoding_perf_v4_raw', "wb") as fp: 
    #     pickle.dump(perf_v4, fp)

    p_v4    =   reorganize_data(perf_v4)
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/decoding_perf_v4C0.8', "wb") as fp: 
        pickle.dump(p_v4, fp)