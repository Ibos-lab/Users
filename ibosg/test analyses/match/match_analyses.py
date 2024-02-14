
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



def moving_average(data:np.ndarray,win:int, step:int=1)-> np.ndarray:
    d_shape=data.shape
    d_avg = np.zeros((d_shape[0],int(np.floor(d_shape[1]/step))))
    d_zsc = np.zeros((d_shape[0],int(np.floor(d_shape[1]/step))))
    count = 0

    for i_step in np.arange(0, d_shape[1]-step, step):
        d_avg[:,count]  =   np.mean(data[:,i_step:i_step+win],axis=1)
        if np.std(d_avg[:,count])!=0:
            d_zsc[:,count]  =   (d_avg[:,count]-np.mean(d_avg[:,count]))/np.std(d_avg[:,count])
        count +=1

    return d_avg, d_zsc



def scale_p(x, out_range=(-1, 1)):
    if np.sum(x>1) >0:
        return
    domain = 0, 1
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2




def compute_roc_auc(group1,group2):
    rng = np.random.default_rng(seed=seed)
    roc_score = []
    p = []
    for n_win in np.arange(group1.shape[1]):
        g1 = group1[:,n_win]
        g2 = group2[:,n_win]

        # Wilcoxon rank-sum 
        p.append(stats.ttest_ind(g1, g2)[1])#stats.ttest_ind

        thresholds = np.unique(np.concatenate([g1,g2]))

        y_g1, y_g2 = np.ones(len(g1)),np.zeros(len(g2))
        score=0.5
        fpr,tpr=[],[]
        for threshold in thresholds:
            g1_y_pred,g2_y_pred = np.zeros(len(g1)),np.zeros(len(g2))
            g1_mask,g2_mask = g1>=threshold,g2>=threshold
            g1_y_pred[g1_mask],g2_y_pred[g2_mask] = 1,1
            tp = sum(np.logical_and(y_g1==1,g1_y_pred==1))
            fn = sum(np.logical_and(y_g1==1,g1_y_pred==0))
            tpr.append(tp/ (tp+fn) )
            fp = sum(np.logical_and(y_g2==0,g2_y_pred==1))
            tn = sum(np.logical_and(y_g2==0,g2_y_pred==0))
            fpr.append(fp/ (fp+tn) )
        if len(fpr) > 1:
            fpr,tpr=np.array(fpr),np.array(tpr)
                      
            score = metrics.auc(fpr[fpr.argsort()],tpr[fpr.argsort()])
        roc_score.append(score)
        
    roc_score = np.array(roc_score)
    roc_score = scale_p(np.round(roc_score,2),out_range=[-1,1])
    return roc_score, p



def ocm(test_data):
    win=100
    step=1
    
    session =   test_data
    # data    =   test_data['data']
    test_data_avg_sp, test_data_z_sp  =   moving_average(data=session['data'],win=win, step=step)
    
    so1c1_to1c1     =   np.where(np.logical_and(session['test id']==11, session['sample id']==11))[0]
    so1c5_to1c1     =   np.where(np.logical_and(session['test id']==11, session['sample id']==15))[0]
    so5c1_to1c1     =   np.where(np.logical_and(session['test id']==11, session['sample id']==51))[0]
    so5c5_to1c1     =   np.where(np.logical_and(session['test id']==11, session['sample id']==55))[0]

    so1c1_to1c5     =   np.where(np.logical_and(session['test id']==15, session['sample id']==11))[0]
    so1c5_to1c5     =   np.where(np.logical_and(session['test id']==15, session['sample id']==15))[0]
    so5c1_to1c5     =   np.where(np.logical_and(session['test id']==15, session['sample id']==51))[0]
    so5c5_to1c5     =   np.where(np.logical_and(session['test id']==15, session['sample id']==55))[0]

    so1c1_to5c1     =   np.where(np.logical_and(session['test id']==51, session['sample id']==11))[0]
    so1c5_to5c1     =   np.where(np.logical_and(session['test id']==51, session['sample id']==15))[0]
    so5c1_to5c1     =   np.where(np.logical_and(session['test id']==51, session['sample id']==51))[0]
    so5c5_to5c1     =   np.where(np.logical_and(session['test id']==51, session['sample id']==55))[0]
    
    so1c1_to5c5     =   np.where(np.logical_and(session['test id']==55, session['sample id']==11))[0]
    so1c5_to5c5     =   np.where(np.logical_and(session['test id']==55, session['sample id']==15))[0]
    so5c1_to5c5     =   np.where(np.logical_and(session['test id']==55, session['sample id']==51))[0]
    so5c5_to5c5     =   np.where(np.logical_and(session['test id']==55, session['sample id']==55))[0]
    
    # test o1 vs o5
    vis_to1     =   np.sort(np.concatenate([so1c1_to1c1, so1c5_to1c1, so5c1_to1c1, so5c5_to1c1,
                                            so1c1_to1c5, so1c5_to1c5, so5c1_to1c5, so5c5_to1c5]))

    vis_to5     =   np.sort(np.concatenate([so1c1_to5c1, so1c5_to5c1, so5c1_to5c1, so5c5_to5c1,
                                            so1c1_to5c5, so1c5_to5c5, so5c1_to5c5, so5c5_to5c5]))

    vis_tc1     =   np.sort(np.concatenate([so1c1_to1c1, so1c5_to1c1, so5c1_to1c1, so5c5_to1c1,
                                            so1c1_to5c1, so1c5_to5c1, so5c1_to5c1, so5c5_to5c1]))

    vis_tc5     =   np.sort(np.concatenate([so1c1_to1c5, so1c5_to1c5, so5c1_to1c5, so5c5_to1c5,
                                            so1c1_to5c5, so1c5_to5c5, so5c1_to5c5, so5c5_to5c5]))

    match_trials    =   np.sort(np.concatenate([so1c1_to1c1, so1c5_to1c5, so5c1_to5c1, so5c5_to5c1]))
    nomatch_trials  =   np.sort(np.concatenate([so1c1_to1c5, so1c1_to5c1,so1c1_to5c5,
                                                so1c5_to1c1, so1c5_to5c1,so1c1_to5c5,
                                                so5c1_to1c1, so5c1_to1c5,so5c1_to5c5,
                                                so5c5_to1c1, so5c5_to1c5,so5c5_to5c1]))
    
    o_test, p_o_test  =   compute_roc_auc(test_data_avg_sp[vis_to1, :], test_data_avg_sp[vis_to5, :])
    c_test, p_c_test  =   compute_roc_auc(test_data_avg_sp[vis_tc1, :], test_data_avg_sp[vis_tc5, :])
    m_test, p_m_test  =   compute_roc_auc(test_data_avg_sp[match_trials, :], test_data_avg_sp[nomatch_trials, :])

    return{'data averaged'  :   test_data_avg_sp, 
           'data zscored'   :   test_data_z_sp, 
           'roc values o'   :   o_test,
           'p values o'     :   p_o_test,
           'roc values c'   :   c_test,
           'p values c'     :   p_c_test,
           'roc values m'   :   m_test,
           'p values m'     :   p_m_test}



def reorganize_data(ocm_result):
    avg_data    =   []
    zsc_data    =   []

    roc_v_m    =   np.empty([len(ocm_result), 800])
    roc_v_o    =   np.empty([len(ocm_result), 800])
    roc_v_c    =   np.empty([len(ocm_result), 800])
    roc_p_m    =   np.empty([len(ocm_result), 800])
    roc_p_o    =   np.empty([len(ocm_result), 800])
    roc_p_c    =   np.empty([len(ocm_result), 800])

    for i in range(len(ocm_result)):
        avg_data.append(ocm_result[i]['data averaged'])
        zsc_data.append(ocm_result[i]['data zscored'])

        roc_v_m[i,:]=ocm_result[i]['roc values m']
        roc_v_o[i,:]=ocm_result[i]['roc values o']
        roc_v_c[i,:]=ocm_result[i]['roc values c']
        roc_p_m[i,:]=ocm_result[i]['p values m']
        roc_p_o[i,:]=ocm_result[i]['p values o']
        roc_p_c[i,:]=ocm_result[i]['p values c']

    return{'averaged data': avg_data,
           'zscored data': zsc_data,
           'roc values m': roc_v_m,
           'roc values o': roc_v_o,
           'roc values c': roc_v_c,
           'p values m': roc_p_m,
           'p values o': roc_p_o,
           'p values c': roc_p_c}


with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/LIP_test', 'rb') as handle:
    test_lip = pickle.load(handle)

with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/PFC_test', 'rb') as handle:
    test_pfc = pickle.load(handle)
    
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/v4_test', 'rb') as handle:
    test_v4 = pickle.load(handle)


# model=  SVC(kernel='linear',C=1, decision_function_shape='ovr',gamma='auto',degree=1)

numcells=len(test_lip)
ocm_result    =   Parallel(n_jobs = -1)(delayed(ocm)(cell) for cell in tqdm(test_lip[:numcells]))
lip_ocm_data=reorganize_data(ocm_result)
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/lip_roc_test', "wb") as fp: 
    pickle.dump(lip_ocm_data, fp)

numcells=len(test_pfc)
ocm_result    =   Parallel(n_jobs = -1)(delayed(ocm)(cell) for cell in tqdm(test_pfc[:numcells]))
pfc_ocm_data=reorganize_data(ocm_result)
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/pfc_roc_test', "wb") as fp: 
    pickle.dump(pfc_ocm_data, fp)

numcells=len(test_v4)
ocm_result    =   Parallel(n_jobs = -1)(delayed(ocm)(cell) for cell in tqdm(test_v4[:numcells]))
v4_ocm_data=reorganize_data(ocm_result)
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/v4_roc_test', "wb") as fp: 
    pickle.dump(v4_ocm_data, fp)