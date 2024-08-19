
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




def perm_single_trials(acti):    
    scrumbled  =   np.random.permutation(acti)
    diff   =   acti-scrumbled
    return diff


def trial_latency(test_data):
    values=[11,15,51,55]
    for test in values:
        globals()['stim_vis_'+ str(test) +'_in']    =   basic_cond(test_data=test_data, sample_value=-1, test_value=test, position=1)
        globals()['stim_vis_'+ str(test)+'_out']    =   basic_cond(test_data=test_data, sample_value=-1, test_value=test, position=-1)
        for sample in values:
            globals()['s_in_'+str(sample)+'_t_'+ str(test)]  =   basic_cond(test_data=test_data, sample_value=sample, test_value=test, position=1)
            globals()['s_out_'+str(sample)+'_t_'+ str(test)] =   basic_cond(test_data=test_data, sample_value=sample, test_value=test, position=-1)


    stim_match_in   =   np.sort(np.concatenate([s_in_11_t_11, s_in_15_t_15, s_in_51_t_51, s_in_55_t_55]))
    stim_match_out  =   np.sort(np.concatenate([s_out_11_t_11, s_out_15_t_15, s_out_51_t_51, s_out_55_t_55]))

    rts_in     =   test_data['reaction times'][stim_match_in]
    rts_out    =   test_data['reaction times'][stim_match_out]

    acti_match_in   =   test_data['averaged data'][stim_match_in,:-100]
    diff_in            =   np.empty([1000, acti_match_in.shape[1]],dtype='single')
    sig_in             =   np.empty([acti_match_in.shape[0], acti_match_in.shape[1]],dtype='single')
    lat_in             =   np.empty(acti_match_in.shape[0])*np.nan

    acti_match_out  =   test_data['averaged data'][stim_match_out,:-100]
    diff_out        =   np.empty([1000,  acti_match_out.shape[1]],dtype='single')
    sig_out         =   np.empty([acti_match_out.shape[0], acti_match_out.shape[1]],dtype='single')
    lat_out         =   np.empty(acti_match_out.shape[0])*np.nan
    
    # define a latency for each trial using time permutation
    m=np.max([acti_match_in.shape[0], acti_match_out.shape[0]])  

    for trial in range(m):
        if trial<acti_match_in.shape[0] and trial<acti_match_out.shape[0]:
            for it in range(1000):
            
                scrumbled_in  =   np.random.permutation(acti_match_in[trial,:])
                diff_in[it,:]    =   acti_match_in[trial,:]-scrumbled_in

                scrumbled_out  =   np.random.permutation(acti_match_out[trial,:])
                diff_out[it,:]    =   acti_match_out[trial,:]-scrumbled_out
            
            sig_in[trial,:]=np.sum(diff_in>0, axis=0)
            sig_out[trial,:]=np.sum(diff_out>0, axis=0)

            if np.sum(sig_in[trial,:]>950)>0:
                lat_in[trial]=np.where(sig_in[trial,:]>950)[0][0]
            if np.sum(sig_out[trial,:]>950)>0:
                lat_out[trial]=np.where(sig_out[trial,:]>950)[0][0]


        if trial>=acti_match_in.shape[0] and trial<acti_match_out.shape[0]:
            for it in range(1000):    
                scrumbled_out  =   np.random.permutation(acti_match_out[trial,:])
                diff_out[it,:]    =   acti_match_out[trial,:]-scrumbled_out
            
            sig_out[trial,:]=np.sum(diff_out>0, axis=0)
            if np.sum(sig_out[trial,:]>950)>0:
                lat_out[trial]=np.where(sig_out[trial,:]>950)[0][0]

        if trial<acti_match_in.shape[0] and trial>=acti_match_out.shape[0]:
            for it in range(1000): 
                scrumbled_in  =   np.random.permutation(acti_match_in[trial,:])
                diff_in[it,:]    =   acti_match_in[trial,:]-scrumbled_in
            
            sig_in[trial,:]=np.sum(diff_in>0, axis=0)
            if np.sum(sig_in[trial,:]>950)>0:
                lat_in[trial]=np.where(sig_in[trial,:]>950)[0][0]
            
    lat_sel_in=lat_in[np.logical_and(lat_in>0, rts_in>0)]
    rts_sel_in=rts_in[np.logical_and(lat_in>0, rts_in>0)]

    if np.sum(len(lat_sel_in>0))>10:
        rin, pin= stats.pearsonr(lat_sel_in, rts_sel_in)
    else:
        rin, pin= [np.nan, np.nan]

    lat_sel_out=lat_out[np.logical_and(lat_out>0, rts_out>0)]
    rts_sel_out=rts_out[np.logical_and(lat_out>0, rts_out>0)]
    if np.sum(len(lat_sel_out>0))>10:
        rout, pout= stats.pearsonr(lat_sel_out, rts_sel_out)        
    else:
        rout, pout= [np.nan, np.nan]

    # results=[rin,pin,rout,pout]
    results={"name": test_data['name'],
             "corr coef match in": rin,
             "p value match in": pin,
             "corr coef match out": rout,
             "p value match out": pout}
    
    return results


with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_test_avg_w50_s1m300p500', 'rb') as handle:
    test_lip_avg = pickle.load(handle)

with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_test_avg_w50_s1m300p500', 'rb') as handle:
    test_pfc_avg = pickle.load(handle)


with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_test_avg_50_s1m300p500', 'rb') as handle:
    test_v4_avg = pickle.load(handle)


def basic_cond(test_data, sample_value, test_value, position):
    if sample_value>=0:
        result=np.where(np.logical_and(np.logical_and(test_data['sample id']==sample_value, test_data['test id']==test_value), test_data['sample pos']==position))[0]
    else:
        result=np.where(np.logical_and(np.logical_and(test_data['sample id']!=0, test_data['test id']==test_value), test_data['sample pos']==position))[0]
    return result



rtcor_lip    =   Parallel(n_jobs = -1)(delayed(trial_latency)(test_lip_avg[cell]) for cell in range(len(test_lip_avg)))
# names=list(test_lip_avg[i]['name'] for i in range(len(test_lip_avg)))
# for i in range(len(rtcor_lip)):
#     nam=rtcor_lip[i]['name']
#     ind=names.index(nam)
#     Coef[ind]['corr coef match in']     =   rtcor_lip[i]['corr coef match in']
#     Coef[ind]['corr pvalue match in']   =   rtcor_lip[i]['p value match in']
#     Coef[ind]['corr coef match out']    =   rtcor_lip[i]['corr coef match out']
#     Coef[ind]['corr pvalue match out']  =   rtcor_lip[i]['p value match out']

print('saving lip')
with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_test_avg_w50_s1m300p500_corrcoef", "wb") as fp: 
    pickle.dump(rtcor_lip, fp)

rtcor_pfc    =   Parallel(n_jobs = -1)(delayed(trial_latency)(test_pfc_avg[cell]) for cell in range(len(test_pfc_avg)))
# names=list(test_pfc_avg[i]['name'] for i in range(len(test_pfc_avg)))
# for i in range(len(rtcor_pfc)):
#     nam=rtcor_pfc[i]['name']
#     ind=names.index(nam)
#     test_pfc_avg[ind]['corr coef match in']     =   rtcor_pfc[i]['corr coef match in']
#     test_pfc_avg[ind]['corr pvalue match in']   =   rtcor_pfc[i]['p value match in']
#     test_pfc_avg[ind]['corr coef match out']    =   rtcor_pfc[i]['corr coef match out']
#     test_pfc_avg[ind]['corr pvalue match out']  =   rtcor_pfc[i]['p value match out']

print('saving pfc')
with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_test_avg_w50_s1m300p500_corrcoef", "wb") as fp: 
    pickle.dump(rtcor_pfc, fp)
    
rtcor_v4    =   Parallel(n_jobs = -1)(delayed(trial_latency)(test_v4_avg[cell]) for cell in range(len(test_v4_avg)))
# names=list(test_v4_avg[i]['name'] for i in range(len(test_v4_avg)))
# for i in range(len(rtcor_v4)):
#     nam=rtcor_v4[i]['name']
#     ind=names.index(nam)
#     test_v4_avg[ind]['corr coef match in']      =   rtcor_v4[i]['corr coef match in']
#     test_v4_avg[ind]['corr pvalue match in']    =   rtcor_v4[i]['p value match in']
#     test_v4_avg[ind]['corr coef match out']     =   rtcor_v4[i]['corr coef match out']
#     test_v4_avg[ind]['corr pvalue match out']   =   rtcor_v4[i]['p value match out']

print('saving v4')
with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_test_avg_w50_s1m300p500_corrcoef", "wb") as fp: 
    pickle.dump(rtcor_v4, fp)