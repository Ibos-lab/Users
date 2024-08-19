
import numpy as np
import numpy.matlib as mt
import h5py
import math 
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import align_trials
from ephysvibe.task import task_constants

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import signal,stats

import os 
from matplotlib import cm
from matplotlib import pyplot as plt
import glob
from pathlib import Path
import pickle

from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import permutations
seed = 2023



def pick_train_test_trials(trials, train_ratio):
    tmp     =   np.random.permutation(trials)
    train   =   tmp[:int((trials.shape[0]*train_ratio))]
    test    =   tmp[int((trials.shape[0]*train_ratio)):]

    train_rep   =   mt.repmat(train, test.shape[0],1)
    test_rep    =   mt.repmat(test, train.shape[0],1)
    if np.where(test_rep-np.transpose(train_rep)==0)[0].shape[0]!=0:
        print("Warnings: similar trials for training and testing")

    return train, test

def even_trials(test_data, match, no_match):

    pos_match       =   test_data['test num'][match]
    
    nomatch_test_num    =   test_data['test num'][no_match]

    num_nomatch_0   =   np.sum(nomatch_test_num==0)
    proportion_nomatch_0    =   num_nomatch_0/no_match.shape[0]
    tmp_match_0  =   np.random.permutation(match[pos_match==0])
    
    num_nomatch_1   =   np.sum(nomatch_test_num==1)
    proportion_nomatch_1    =   num_nomatch_1/no_match.shape[0]
    tmp_match_1  =   np.random.permutation(match[pos_match==1])
    
    num_nomatch_2   =   np.sum(nomatch_test_num==2)
    proportion_nomatch_2    =   num_nomatch_2/no_match.shape[0]
    tmp_match_2  =   np.random.permutation(match[pos_match==2])
    
    
    num_nomatch_3   =   np.sum(nomatch_test_num==3)
    proportion_nomatch_3    =   num_nomatch_3/no_match.shape[0]
    tmp_match_3  =   np.random.permutation(match[pos_match==3])
    
    
    num_nomatch_4   =   np.sum(nomatch_test_num==4)
    proportion_nomatch_4    =   num_nomatch_4/no_match.shape[0]
    tmp_match_4  =   np.random.permutation(match[pos_match==4])
    

    # tmp_match_n0 =   np.random.permutation(match[pos_match>0])

    num0    =   np.round(tmp_match_0.shape[0]*proportion_nomatch_0)
    num1    =   np.round(tmp_match_1.shape[0]*proportion_nomatch_1)
    num2    =   np.round(tmp_match_2.shape[0]*proportion_nomatch_2)
    num3    =   np.round(tmp_match_3.shape[0]*proportion_nomatch_3)
    num4    =   np.round(tmp_match_4.shape[0]*proportion_nomatch_4)
    # numn0   =   np.round(tmp_match_n0.shape[0]*(1-proportion_nomatch_0))

    match_trials    =   np.concatenate([tmp_match_0[:int(num0)],tmp_match_1[:int(num1)],tmp_match_2[:int(num2)],tmp_match_3[:int(num3)],tmp_match_4[:int(num4)] ])

    return match_trials

def basic_cond(test_data, sample_value, test_value, position):
    if sample_value>=0:
        result=np.where(np.logical_and(np.logical_and(test_data['sample id']==sample_value, test_data['test id']==test_value), test_data['sample pos']==position))[0]
    else:
        result=np.where(np.logical_and(np.logical_and(test_data['sample id']!=0, test_data['test id']==test_value), test_data['sample pos']==position))[0]
    return result
def get_cond(test_data):
    # basic conditions
    values=[11,15,51,55]
    
    for test in values:
        globals()['trials_s_in_t_'+ str(test)]   =   basic_cond(test_data=test_data, sample_value=-1, test_value=test, position=1)
        globals()['trials_s_out_t_'+ str(test)]  =   basic_cond(test_data=test_data, sample_value=-1, test_value=test, position=-1)
        for sample in values:
            globals()['s_in_'+str(sample)+'_t_'+ str(test)]  =   basic_cond(test_data=test_data, sample_value=sample, test_value=test, position=1)
            globals()['s_out_'+str(sample)+'_t_'+ str(test)] =   basic_cond(test_data=test_data, sample_value=sample, test_value=test, position=-1)

    s_in_11_nomatch     =    np.concatenate([s_in_11_t_15, s_in_11_t_51, s_in_11_t_55])
    s_in_15_nomatch     =    np.concatenate([s_in_15_t_11, s_in_15_t_51, s_in_15_t_55])
    s_in_51_nomatch     =    np.concatenate([s_in_51_t_11, s_in_51_t_15, s_in_51_t_55])
    s_in_55_nomatch     =    np.concatenate([s_in_55_t_11, s_in_55_t_15, s_in_55_t_51])

    num_s_in_11_nomatch    =   s_in_11_nomatch.shape[0]
    num_s_in_15_nomatch    =   s_in_15_nomatch.shape[0]
    num_s_in_51_nomatch    =   s_in_51_nomatch.shape[0]
    num_s_in_55_nomatch    =   s_in_55_nomatch.shape[0]

    s_in_11_t_11b   =   even_trials(test_data=test_data, match=s_in_11_t_11, no_match=s_in_11_nomatch)
    s_in_15_t_15b   =   even_trials(test_data=test_data, match=s_in_15_t_15, no_match=s_in_15_nomatch)
    s_in_51_t_51b   =   even_trials(test_data=test_data, match=s_in_51_t_51, no_match=s_in_51_nomatch)
    s_in_55_t_55b   =   even_trials(test_data=test_data, match=s_in_55_t_55, no_match=s_in_55_nomatch)

    num_s_in_11_match    =   s_in_11_t_11b.shape[0]
    num_s_in_15_match    =   s_in_15_t_15b.shape[0]
    num_s_in_51_match    =   s_in_51_t_51b.shape[0]
    num_s_in_55_match    =   s_in_55_t_55b.shape[0]    

    num_11  =   np.min([num_s_in_11_match, num_s_in_11_nomatch])
    num_15  =   np.min([num_s_in_15_match, num_s_in_15_nomatch])
    num_51  =   np.min([num_s_in_51_match, num_s_in_51_nomatch])
    num_55  =   np.min([num_s_in_55_match, num_s_in_55_nomatch])

    s_in_11_nomatch     =    np.random.permutation(s_in_11_nomatch)[:num_11]
    s_in_15_nomatch     =    np.random.permutation(s_in_15_nomatch)[:num_15]
    s_in_51_nomatch     =    np.random.permutation(s_in_51_nomatch)[:num_51]
    s_in_55_nomatch     =    np.random.permutation(s_in_55_nomatch)[:num_55]

    s_in_11_t_11b   =   s_in_11_t_11b[:num_11]
    s_in_15_t_15b   =   s_in_15_t_15b[:num_15]
    s_in_51_t_51b   =   s_in_51_t_51b[:num_51]
    s_in_55_t_55b   =   s_in_55_t_55b[:num_55]

    trials_s_in_t_match     =   np.concatenate([s_in_11_t_11b, s_in_15_t_15b, s_in_51_t_51b, s_in_55_t_55b])
    trials_s_in_t_nomatch   =   np.concatenate([s_in_11_nomatch, s_in_15_nomatch, s_in_51_nomatch, s_in_55_nomatch])

    s_out_11_nomatch     =    np.concatenate([s_out_11_t_15, s_out_11_t_51, s_out_11_t_55])
    s_out_15_nomatch     =    np.concatenate([s_out_15_t_11, s_out_15_t_51, s_out_15_t_55])
    s_out_51_nomatch     =    np.concatenate([s_out_51_t_11, s_out_51_t_15, s_out_51_t_55])
    s_out_55_nomatch     =    np.concatenate([s_out_55_t_11, s_out_55_t_15, s_out_55_t_51])

    num_s_out_11_nomatch    =   s_out_11_nomatch.shape[0]
    num_s_out_15_nomatch    =   s_out_15_nomatch.shape[0]
    num_s_out_51_nomatch    =   s_out_51_nomatch.shape[0]
    num_s_out_55_nomatch    =   s_out_55_nomatch.shape[0]

    s_out_11_t_11b   =   even_trials(test_data=test_data, match=s_out_11_t_11, no_match=s_out_11_nomatch)
    s_out_15_t_15b   =   even_trials(test_data=test_data, match=s_out_15_t_15, no_match=s_out_15_nomatch)
    s_out_51_t_51b   =   even_trials(test_data=test_data, match=s_out_51_t_51, no_match=s_out_51_nomatch)
    s_out_55_t_55b   =   even_trials(test_data=test_data, match=s_out_55_t_55, no_match=s_out_55_nomatch)

    num_s_out_11_match    =   s_out_11_t_11b.shape[0]
    num_s_out_15_match    =   s_out_15_t_15b.shape[0]
    num_s_out_51_match    =   s_out_51_t_51b.shape[0]
    num_s_out_55_match    =   s_out_55_t_55b.shape[0]    

    num_11  =   np.min([num_s_out_11_match, num_s_out_11_nomatch])
    num_15  =   np.min([num_s_out_15_match, num_s_out_15_nomatch])
    num_51  =   np.min([num_s_out_51_match, num_s_out_51_nomatch])
    num_55  =   np.min([num_s_out_55_match, num_s_out_55_nomatch])

    s_out_11_nomatch     =    np.random.permutation(s_out_11_nomatch)[:num_11]
    s_out_15_nomatch     =    np.random.permutation(s_out_15_nomatch)[:num_15]
    s_out_51_nomatch     =    np.random.permutation(s_out_51_nomatch)[:num_51]
    s_out_55_nomatch     =    np.random.permutation(s_out_55_nomatch)[:num_55]

    s_out_11_t_11b   =   s_out_11_t_11b[:num_11]
    s_out_15_t_15b   =   s_out_15_t_15b[:num_15]
    s_out_51_t_51b   =   s_out_51_t_51b[:num_51]
    s_out_55_t_55b   =   s_out_55_t_55b[:num_55]

    trials_s_out_t_match     =   np.concatenate([s_out_11_t_11b, s_out_15_t_15b, s_out_51_t_51b, s_out_55_t_55b])
    trials_s_out_t_nomatch   =   np.concatenate([s_out_11_nomatch, s_out_15_nomatch, s_out_51_nomatch, s_out_55_nomatch])

    trials_s_bar_release    =   np.concatenate([trials_s_in_t_match, trials_s_out_t_match])
    trials_s_hold_bar       =   np.concatenate([trials_s_in_t_nomatch, trials_s_out_t_nomatch])

    return {'num_s_11_nomatch': num_s_in_11_nomatch,
            'num_s_15_nomatch': num_s_in_15_nomatch,
            'num_s_51_nomatch': num_s_in_51_nomatch,
            'num_s_55_nomatch': num_s_in_55_nomatch,
            's in t 11': trials_s_in_t_11,
            's in t 15': trials_s_in_t_15,
            's in t 51': trials_s_in_t_51,
            's in t 55': trials_s_in_t_55,
            's out t 11': trials_s_out_t_11,
            's out t 15': trials_s_out_t_15,
            's out t 51': trials_s_out_t_51,
            's out t 55': trials_s_out_t_55,
            's in t match': trials_s_in_t_match,
            's in t nomatch': trials_s_in_t_nomatch,
            's out t match': trials_s_out_t_match,
            's out t nomatch': trials_s_out_t_nomatch,
            'bar release': trials_s_bar_release, 
            'hold bar': trials_s_hold_bar}


model=  SVC(kernel='linear',C=1,decision_function_shape='ovr',gamma='auto',degree=1)


def decode_test_par(dat):
    num_train   =   40
    num_test    =   10
    test_train_ratio    =   1-num_test/num_train

    cond=[]
    sel_cells=[]    
    for i in range(len(dat)):
        tmp=get_cond(dat[i])  
        cond.append(tmp)
        # if np.min(np.concatenate([tmp['s in t 11'].shape, tmp['s in t 15'].shape, 
        #                           tmp['s in t 51'].shape, tmp['s in t 55'].shape, 
        #                           tmp['s in t match'].shape, tmp['s in t nomatch'].shape,
        #                             tmp['s out t 11'].shape, tmp['s in t 15'].shape,
        #                             tmp['s out t 51'].shape, tmp['s out t 55'].shape, 
        #                             tmp['s out t match'].shape, tmp['s out t nomatch'].shape]))>50:
        if np.min([tmp['num_s_11_nomatch'],tmp['num_s_15_nomatch'],tmp['num_s_51_nomatch'],tmp['num_s_55_nomatch']])>=20:
            sel_cells.append(i)
            

    num_cells   =   len(sel_cells)
    sel_cells   =   np.array(sel_cells)[np.random.permutation(num_cells)][:300]

    names       =   list(dat[i]['name'] for i in sel_cells)
    if 'averaged data' in dat[i].keys():
        datas       =   list(dat[i]['averaged data'] for i in sel_cells)
    elif 'zscored data' in dat[i].keys():
        datas       =   list(dat[i]['zscored data'] for i in sel_cells)

    cond        =   list(cond[i] for i in sel_cells)

    data_train_id_in    =   np.empty([datas[0].shape[1], num_train*4, num_cells])
    data_train_id_out   =   np.empty([datas[0].shape[1], num_train*4, num_cells])
    data_train_match_in =   np.empty([datas[0].shape[1], num_train*2, num_cells])
    data_train_match_out=   np.empty([datas[0].shape[1], num_train*2, num_cells])
    data_train_bar      =   np.empty([datas[0].shape[1], num_train*2, num_cells])

    data_test_id_in    =   np.empty([datas[0].shape[1], num_test*4, num_cells])
    data_test_id_out   =   np.empty([datas[0].shape[1], num_test*4, num_cells])
    data_test_match_in =   np.empty([datas[0].shape[1], num_test*2, num_cells])
    data_test_match_out=   np.empty([datas[0].shape[1], num_test*2, num_cells])
    data_test_bar      =   np.empty([datas[0].shape[1], num_test*2, num_cells])

    y_train_id  =   np.concatenate([np.zeros(num_train), np.ones(num_train), np.zeros(num_train)+2, np.ones(num_train)+2])
    y_test_id   =   np.concatenate([np.zeros(num_test), np.ones(num_test), np.zeros(num_test)+2, np.ones(num_test)+2])

    y_train_match=   np.concatenate([np.zeros(num_train), np.ones(num_train)])
    y_test_match =   np.concatenate([np.zeros(num_test), np.ones(num_test)])

    perf_id_in          =   np.empty([len(data_train_id_in)])
    perf_id_out         =   np.empty([len(data_train_id_out)])    
    perf_match_in       =   np.empty([len(data_train_match_in)])
    perf_match_out      =   np.empty([len(data_train_match_out)])
    perf_bar            =   np.empty([len(data_train_bar)])
    perf_match_train_in_test_out=   np.empty([len(data_train_bar)])
    perf_match_train_out_test_in=   np.empty([len(data_train_bar)])

    for cell in range(len(sel_cells)):
        # id trials in
        train_11_in, test_11_in =   pick_train_test_trials(cond[cell]['s in t 11'], test_train_ratio)
        train_15_in, test_15_in =   pick_train_test_trials(cond[cell]['s in t 15'], test_train_ratio)
        train_51_in, test_51_in =   pick_train_test_trials(cond[cell]['s in t 51'], test_train_ratio)
        train_55_in, test_55_in =   pick_train_test_trials(cond[cell]['s in t 55'], test_train_ratio)

        trials_train_id_in      =   np.concatenate([np.random.choice(train_11_in, num_train),
                                                    np.random.choice(train_15_in, num_train),
                                                    np.random.choice(train_51_in, num_train),
                                                    np.random.choice(train_55_in, num_train)])
        
        trials_test_id_in  =   np.concatenate([np.random.choice(test_11_in, num_test),
                                                np.random.choice(test_15_in, num_test),
                                                np.random.choice(test_51_in, num_test),
                                                np.random.choice(test_55_in, num_test),])
        
        # id trials out
        train_11_out, test_11_out   =  pick_train_test_trials(cond[cell]['s out t 11'], test_train_ratio)
        train_15_out, test_15_out   =  pick_train_test_trials(cond[cell]['s out t 15'], test_train_ratio)
        train_51_out, test_51_out   =  pick_train_test_trials(cond[cell]['s out t 51'], test_train_ratio)
        train_55_out, test_55_out   =  pick_train_test_trials(cond[cell]['s out t 55'], test_train_ratio)
        trials_train_id_out =   np.concatenate([np.random.choice(train_11_out, num_train),
                                                np.random.choice(train_15_out, num_train),
                                                np.random.choice(train_51_out, num_train),
                                                np.random.choice(train_55_out, num_train)])
        trials_test_id_out  =   np.concatenate([np.random.choice(test_11_out, num_test),
                                                np.random.choice(test_15_out, num_test),
                                                np.random.choice(test_51_out, num_test),
                                                np.random.choice(test_55_out, num_test),])
        # match trials in
        train_match_in, test_match_in   =   pick_train_test_trials(cond[cell]['s in t match'], test_train_ratio)
        train_nomatch_in, test_nomatch_in =  pick_train_test_trials(cond[cell]['s in t nomatch'], test_train_ratio)
        trials_train_match_in   =   np.concatenate([np.random.choice(train_match_in, num_train),
                                                np.random.choice(train_nomatch_in, num_train)])
        trials_test_match_in    =   np.concatenate([np.random.choice(test_match_in, num_test),
                                                np.random.choice(test_nomatch_in, num_test)])
        
        # match trials out
        train_match_out, test_match_out =  pick_train_test_trials(cond[cell]['s out t match'], test_train_ratio)
        train_nomatch_out, test_nomatch_out =  pick_train_test_trials(cond[cell]['s out t nomatch'], test_train_ratio)
        trials_train_match_out  =   np.concatenate([np.random.choice(train_match_out, num_train),
                                                np.random.choice(train_nomatch_out, num_train)])
        trials_test_match_out   =   np.concatenate([np.random.choice(test_match_out, num_test),
                                                np.random.choice(test_nomatch_out, num_test)])
                
        # bar release
        train_bar, test_bar =  pick_train_test_trials(cond[cell]['bar release'], test_train_ratio)
        train_nobar, test_nobar =  pick_train_test_trials(cond[cell]['hold bar'], test_train_ratio)
        trials_train_bar    =   np.concatenate([np.random.choice(train_bar, num_train),
                                                np.random.choice(train_nobar, num_train)])
        trials_test_bar     =   np.concatenate([np.random.choice(test_bar, num_test),
                                                np.random.choice(test_nobar, num_test)])
                
        # build matrices of  [timestamp, trials, neurons] dimensions to feed to classifiers
        
        data_train_id_in[:, :, cell]        =   np.transpose(datas[cell][trials_train_id_in])
        data_train_id_out[:, :, cell]       =   np.transpose(datas[cell][trials_train_id_out])
        data_train_match_in[:, :, cell]     =   np.transpose(datas[cell][trials_train_match_in])
        data_train_match_out[:, :, cell]    =   np.transpose(datas[cell][trials_train_match_out])   
        data_train_bar[:, :, cell]          =   np.transpose(datas[cell][trials_train_bar])
        
        data_test_id_in[:, :, cell]         =   np.transpose(datas[cell][trials_test_id_in])
        data_test_id_out[:, :, cell]        =   np.transpose(datas[cell][trials_test_id_out])
        data_test_match_in[:, :, cell]      =   np.transpose(datas[cell][trials_test_match_in])
        data_test_match_out[:, :, cell]     =   np.transpose(datas[cell][trials_test_match_out])        
        data_test_bar[:, :, cell]           =   np.transpose(datas[cell][trials_test_bar])

    for time_train in range(data_train_id_in.shape[0]):
        model.fit(data_train_id_in[time_train],y_train_id)
        y_predict = model.predict(data_test_id_in[time_train])
        perf_id_in[time_train]=np.where(y_predict-y_test_id==0)[0].shape[0]

        model.fit(data_train_id_out[time_train],y_train_id)
        y_predict = model.predict(data_test_id_out[time_train])
        perf_id_out[time_train]=np.where(y_predict-y_test_id==0)[0].shape[0]

        model.fit(data_train_match_in[time_train],y_train_match)
        y_predict = model.predict(data_test_match_in[time_train])
        perf_match_in[time_train]=np.where(y_predict-y_test_match==0)[0].shape[0]

        model.fit(data_train_match_out[time_train],y_train_match)
        y_predict = model.predict(data_test_match_out[time_train])
        perf_match_out[time_train]=np.where(y_predict-y_test_match==0)[0].shape[0]

        model.fit(data_train_match_in[time_train],y_train_match)
        y_predict = model.predict(data_test_match_out[time_train])
        perf_match_train_in_test_out[time_train]=np.where(y_predict-y_test_match==0)[0].shape[0]
        
        model.fit(data_train_match_out[time_train],y_train_match)
        y_predict = model.predict(data_test_match_in[time_train])
        perf_match_train_out_test_in[time_train]=np.where(y_predict-y_test_match==0)[0].shape[0]


        model.fit(data_train_bar[time_train],y_train_match)
        y_predict = model.predict(data_test_bar[time_train])
        perf_bar[time_train]=np.where(y_predict-y_test_match==0)[0].shape[0]

    return{'lip num cells': len(sel_cells),
           'perf id in': perf_id_in,
           'perf id out': perf_id_out,
           'perf match in': perf_match_in,
           'perf match out': perf_match_out,
           'perf x match in_out': perf_match_train_in_test_out,
           'perf x match out_in': perf_match_train_out_test_in,
           'perf bar': perf_bar}



with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test//lip_test_zsc_w50_s1m300p500', 'rb') as handle:
    test_lip_avg = pickle.load(handle)
    
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test//pfc_test_zsc_w50_s1m300p500', 'rb') as handle:
    test_pfc_avg = pickle.load(handle)
    
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test//v4_test_zsc_w50_s1m300p500', 'rb') as handle:
    test_v4_avg = pickle.load(handle)

def reorganize_data(data, num_iter):
    p_test_id_in    =   np.empty([num_iter, data[0]['perf id in'].shape[0]])*np.nan
    p_test_id_out   =   np.empty([num_iter, data[0]['perf id out'].shape[0]])*np.nan
    p_test_match_in =   np.empty([num_iter, data[0]['perf match in'].shape[0]])*np.nan
    p_test_match_out=   np.empty([num_iter, data[0]['perf match out'].shape[0]])*np.nan
    p_test_xmatch_inout =   np.empty([num_iter, data[0]['perf x match in_out'].shape[0]])*np.nan
    p_test_xmatch_outin =   np.empty([num_iter, data[0]['perf x match out_in'].shape[0]])*np.nan
    p_test_bar      =   np.empty([num_iter, data[0]['perf bar'].shape[0]])*np.nan

    for i in range(num_iter):
        p_test_id_in[i,:]       =   data[i]['perf id in']
        p_test_id_out[i,:]      =   data[i]['perf id out']
        p_test_match_in[i,:]    =   data[i]['perf match in']
        p_test_match_out[i,:]   =   data[i]['perf match out']
        p_test_xmatch_inout[i,:]    =   data[i]['perf x match in_out']
        p_test_xmatch_outin[i,:]    =   data[i]['perf x match out_in']
        p_test_bar[i,:]         =   data[i]['perf bar']
    
    return {'perf id in':p_test_id_in,
            'perf id out':p_test_id_out,
            'perf match in':p_test_match_in,
            'perf match out':p_test_match_out,
            'perf xmatch inout':p_test_xmatch_inout,
            'perf xmatch outin':p_test_xmatch_outin,
            'perf bar':p_test_bar}
    


do_lip=1
do_pfc=1
do_v4=1
num_iter=1000



if do_lip==1:
        # perf    =   Parallel(n_jobs = -1)(delayed(decode_test_par)(test_lip_avg) for cell in tqdm(range(num_iter)))
        # perf_lip=reorganize_data(perf, num_iter)
    perf=[]
    for cell in tqdm(range(num_iter)):
        r=decode_test_par(test_lip_avg)
        perf.append(r)
        r=[]
    perf_lip=reorganize_data(perf, num_iter)
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/decoding_perf_lip_C1_s1_w50', "wb") as fp: 
        pickle.dump(perf_lip, fp)
    perf_lip=[]
if do_pfc==1:
    
        # perf    =   Parallel(n_jobs = -1)(delayed(decode_test_par)(test_pfc_avg) for cell in tqdm(range(num_iter)))
    perf=[]
    for cell in tqdm(range(num_iter)):
        r=decode_test_par(test_pfc_avg)
        perf.append(r)
        r=[]
    
    perf_pfc=reorganize_data(perf, num_iter)
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/decoding_perf_pfc_C1_s1_w50', "wb") as fp: 
        pickle.dump(perf_pfc, fp)
    perf_pfc=[]
if do_v4==1:
        # perf    =   Parallel(n_jobs = -1)(delayed(decode_test_par)(test_v4_avg) for cell in tqdm(range(num_iter)))
    perf=[]
    for cell in tqdm(range(num_iter)):
        r=decode_test_par(test_v4_avg)
        perf.append(r)
        r=[]
    
    perf_decoding_v4=reorganize_data(perf, num_iter)
    with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/decoding_perf_v4_C1_s1_w50', "wb") as fp: 
        pickle.dump(perf_decoding_v4, fp)
    perf_decoding_v4=[]