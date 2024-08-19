
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
        p.append(stats.ranksums(g1, g2)[1])#stats.ttest_ind

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

def basic_cond(test_data, sample_value, test_value, position):
    if sample_value>=0:
        result=np.where(np.logical_and(np.logical_and(test_data['sample id']==sample_value, test_data['test id']==test_value), test_data['sample pos']==position))[0]
    else:
        result=np.where(np.logical_and(np.logical_and(test_data['sample id']!=0, test_data['test id']==test_value), test_data['sample pos']==position))[0]
    return result

# def get_cond(test_data):
#     # basic conditions
    
#     s_in_11_nomatch     =    np.concatenate([s_in_11_t_15, s_in_11_t_51, s_in_11_t_55])
#     s_in_15_nomatch     =    np.concatenate([s_in_15_t_11, s_in_15_t_51, s_in_15_t_55])
#     s_in_51_nomatch     =    np.concatenate([s_in_51_t_11, s_in_51_t_15, s_in_51_t_55])
#     s_in_55_nomatch     =    np.concatenate([s_in_55_t_11, s_in_55_t_15, s_in_55_t_51])

#     num_s_in_11_nomatch    =   s_in_11_nomatch.shape[0]
#     num_s_in_15_nomatch    =   s_in_15_nomatch.shape[0]
#     num_s_in_51_nomatch    =   s_in_51_nomatch.shape[0]
#     num_s_in_55_nomatch    =   s_in_55_nomatch.shape[0]

#     s_in_11_t_11b   =   s_in_11_t_11#even_trials(test_data=test_data, match=s_in_11_t_11, no_match=s_in_11_nomatch)
#     s_in_15_t_15b   =   s_in_15_t_15#even_trials(test_data=test_data, match=s_in_15_t_15, no_match=s_in_15_nomatch)
#     s_in_51_t_51b   =   s_in_51_t_51#even_trials(test_data=test_data, match=s_in_51_t_51, no_match=s_in_51_nomatch)
#     s_in_55_t_55b   =   s_in_55_t_55#even_trials(test_data=test_data, match=s_in_55_t_55, no_match=s_in_55_nomatch)

#     num_s_in_11_match    =   s_in_11_t_11b.shape[0]
#     num_s_in_15_match    =   s_in_15_t_15b.shape[0]
#     num_s_in_51_match    =   s_in_51_t_51b.shape[0]
#     num_s_in_55_match    =   s_in_55_t_55b.shape[0]    

#     # num_11  =   np.min([num_s_in_11_match, num_s_in_11_nomatch])
#     # num_15  =   np.min([num_s_in_15_match, num_s_in_15_nomatch])
#     # num_51  =   np.min([num_s_in_51_match, num_s_in_51_nomatch])
#     # num_55  =   np.min([num_s_in_55_match, num_s_in_55_nomatch])

#     # s_in_11_nomatch     =    np.random.permutation(s_in_11_nomatch)[:num_11]
#     # s_in_15_nomatch     =    np.random.permutation(s_in_15_nomatch)[:num_15]
#     # s_in_51_nomatch     =    np.random.permutation(s_in_51_nomatch)[:num_51]
#     # s_in_55_nomatch     =    np.random.permutation(s_in_55_nomatch)[:num_55]

#     # s_in_11_t_11b   =   s_in_11_t_11b[:num_11]
#     # s_in_15_t_15b   =   s_in_15_t_15b[:num_15]
#     # s_in_51_t_51b   =   s_in_51_t_51b[:num_51]
#     # s_in_55_t_55b   =   s_in_55_t_55b[:num_55]

#     trials_s_in_t_match     =   np.concatenate([s_in_11_t_11b, s_in_15_t_15b, s_in_51_t_51b, s_in_55_t_55b])
#     trials_s_in_t_nomatch   =   np.concatenate([s_in_11_nomatch, s_in_15_nomatch, s_in_51_nomatch, s_in_55_nomatch])

#     s_out_11_nomatch     =    np.concatenate([s_out_11_t_15, s_out_11_t_51, s_out_11_t_55])
#     s_out_15_nomatch     =    np.concatenate([s_out_15_t_11, s_out_15_t_51, s_out_15_t_55])
#     s_out_51_nomatch     =    np.concatenate([s_out_51_t_11, s_out_51_t_15, s_out_51_t_55])
#     s_out_55_nomatch     =    np.concatenate([s_out_55_t_11, s_out_55_t_15, s_out_55_t_51])

#     # num_s_out_11_nomatch    =   s_out_11_nomatch.shape[0]
#     # num_s_out_15_nomatch    =   s_out_15_nomatch.shape[0]
#     # num_s_out_51_nomatch    =   s_out_51_nomatch.shape[0]
#     # num_s_out_55_nomatch    =   s_out_55_nomatch.shape[0]

#     s_out_11_t_11b   =   s_out_11_t_11#even_trials(test_data=test_data, match=s_out_11_t_11, no_match=s_out_11_nomatch)
#     s_out_15_t_15b   =   s_out_15_t_15#even_trials(test_data=test_data, match=s_out_15_t_15, no_match=s_out_15_nomatch)
#     s_out_51_t_51b   =   s_out_51_t_51#even_trials(test_data=test_data, match=s_out_51_t_51, no_match=s_out_51_nomatch)
#     s_out_55_t_55b   =   s_out_55_t_55#even_trials(test_data=test_data, match=s_out_55_t_55, no_match=s_out_55_nomatch)

#     # num_s_out_11_match    =   s_out_11_t_11b.shape[0]
#     # num_s_out_15_match    =   s_out_15_t_15b.shape[0]
#     # num_s_out_51_match    =   s_out_51_t_51b.shape[0]
#     # num_s_out_55_match    =   s_out_55_t_55b.shape[0]    

#     # num_11  =   np.min([num_s_out_11_match, num_s_out_11_nomatch])
#     # num_15  =   np.min([num_s_out_15_match, num_s_out_15_nomatch])
#     # num_51  =   np.min([num_s_out_51_match, num_s_out_51_nomatch])
#     # num_55  =   np.min([num_s_out_55_match, num_s_out_55_nomatch])

#     # s_out_11_nomatch     =    np.random.permutation(s_out_11_nomatch)[:num_11]
#     # s_out_15_nomatch     =    np.random.permutation(s_out_15_nomatch)[:num_15]
#     # s_out_51_nomatch     =    np.random.permutation(s_out_51_nomatch)[:num_51]
#     # s_out_55_nomatch     =    np.random.permutation(s_out_55_nomatch)[:num_55]

#     # s_out_11_t_11b   =   s_out_11_t_11b[:num_11]
#     # s_out_15_t_15b   =   s_out_15_t_15b[:num_15]
#     # s_out_51_t_51b   =   s_out_51_t_51b[:num_51]
#     # s_out_55_t_55b   =   s_out_55_t_55b[:num_55]

#     trials_s_out_t_match     =   np.concatenate([s_out_11_t_11b, s_out_15_t_15b, s_out_51_t_51b, s_out_55_t_55b])
#     trials_s_out_t_nomatch   =   np.concatenate([s_out_11_nomatch, s_out_15_nomatch, s_out_51_nomatch, s_out_55_nomatch])

#     trials_s_bar_release    =   np.concatenate([trials_s_in_t_match, trials_s_out_t_match])
#     trials_s_hold_bar       =   np.concatenate([trials_s_in_t_nomatch, trials_s_out_t_nomatch])

#     return {'name':test_data['name'],
#             'num_s_11_nomatch': num_s_in_11_nomatch,
#             'num_s_15_nomatch': num_s_in_15_nomatch,
#             'num_s_51_nomatch': num_s_in_51_nomatch,
#             'num_s_55_nomatch': num_s_in_55_nomatch,
#             's in t 11': trials_s_in_t_11,
#             's in t 15': trials_s_in_t_15,
#             's in t 51': trials_s_in_t_51,
#             's in t 55': trials_s_in_t_55,
#             's out t 11': trials_s_out_t_11,
#             's out t 15': trials_s_out_t_15,
#             's out t 51': trials_s_out_t_51,
#             's out t 55': trials_s_out_t_55,
#             's in 11 t 11': s_in_11_t_11b,
#             's in 15 t 15': s_in_15_t_15b,
#             's in 51 t 51': s_in_51_t_51b,
#             's in t match': trials_s_in_t_match,
#             's in t nomatch': trials_s_in_t_nomatch,
#             's out t match': trials_s_out_t_match,
#             's out t nomatch': trials_s_out_t_nomatch,
#             'bar release': trials_s_bar_release, 
#             'hold bar': trials_s_hold_bar}

def ocm(test_data):
    if test_data is not None:
        values=[11,15,51,55]
            
        for test in values:
            globals()['stim_vis_'+ str(test) +'_in']    =   basic_cond(test_data=test_data, sample_value=-1, test_value=test, position=1)
            globals()['stim_vis_'+ str(test)+'_out']    =   basic_cond(test_data=test_data, sample_value=-1, test_value=test, position=-1)
            for sample in values:
                globals()['s_in_'+str(sample)+'_t_'+ str(test)] =   basic_cond(test_data=test_data, sample_value=sample, test_value=test, position=1)
                globals()['s_out_'+str(sample)+'_t_'+ str(test)]=   basic_cond(test_data=test_data, sample_value=sample, test_value=test, position=-1)

        stim_match_1155_in      =   np.sort(np.concatenate([s_in_11_t_11, s_in_55_t_55]))
        stim_nomatch_1155_in    =   np.sort(np.concatenate([s_in_15_t_11, s_in_51_t_11,s_in_55_t_11, 
                                                            s_in_11_t_55, s_in_15_t_55,s_in_51_t_55]))
        
        stim_match_1551_in     =   np.sort(np.concatenate([s_in_15_t_15, s_in_51_t_51]))
        stim_nomatch_1551_in   =   np.sort(np.concatenate([s_in_11_t_15, s_in_51_t_15, s_in_55_t_15,
                                                        s_in_11_t_51, s_in_15_t_51,s_in_55_t_51]))

        stim_match_in    =   np.sort(np.concatenate([s_in_11_t_11, s_in_15_t_15, s_in_51_t_51, s_in_55_t_55]))
        stim_nomatch_in  =   np.sort(np.concatenate([s_in_11_t_15, s_in_11_t_51,s_in_11_t_55,
                                                    s_in_15_t_11, s_in_15_t_51,s_in_15_t_55,
                                                    s_in_51_t_11, s_in_51_t_15,s_in_51_t_55,
                                                    s_in_55_t_11, s_in_55_t_15,s_in_55_t_51]))

        roc_vis_1155_zs_in, p_vis_1155_zs_in        =   compute_roc_auc(test_data['zscored data'][stim_vis_11_in, :], test_data['zscored data'][stim_vis_55_in, :])
        roc_match_1155_zs_in, p_match_1155_zs_in    =   compute_roc_auc(test_data['zscored data'][stim_match_1155_in, :], test_data['zscored data'][stim_nomatch_1155_in, :])

        roc_vis_1551_zs_in, p_vis_1551_zs_in        =   compute_roc_auc(test_data['zscored data'][stim_vis_15_in, :], test_data['zscored data'][stim_vis_51_in, :])
        roc_match_1551_zs_in, p_match_1551_zs_in    =   compute_roc_auc(test_data['zscored data'][stim_match_1551_in, :], test_data['zscored data'][stim_nomatch_1551_in, :])

        roc_match_zs_in, p_match_zs_in  =   compute_roc_auc(test_data['zscored data'][stim_match_in, :], test_data['zscored data'][stim_nomatch_in, :])

        # OUT

        stim_match_1155_out      =   np.sort(np.concatenate([s_out_11_t_11, s_out_55_t_55]))
        stim_nomatch_1155_out    =   np.sort(np.concatenate([s_out_15_t_11, s_out_51_t_11,s_out_55_t_11, 
                                                            s_out_11_t_55, s_out_15_t_55,s_out_51_t_55]))
        
        stim_match_1551_out     =   np.sort(np.concatenate([s_out_15_t_15, s_out_51_t_51]))
        stim_nomatch_1551_out   =   np.sort(np.concatenate([s_out_11_t_15, s_out_51_t_15, s_out_55_t_15,
                                                        s_out_11_t_51, s_out_15_t_51,s_out_55_t_51]))

        stim_match_out    =   np.sort(np.concatenate([s_out_11_t_11, s_out_15_t_15, s_out_51_t_51, s_out_55_t_55]))
        stim_nomatch_out  =   np.sort(np.concatenate([s_out_11_t_15, s_out_11_t_51,s_out_11_t_55,
                                                    s_out_15_t_11, s_out_15_t_51,s_out_15_t_55,
                                                    s_out_51_t_11, s_out_51_t_15,s_out_51_t_55,
                                                    s_out_55_t_11, s_out_55_t_15,s_out_55_t_51]))

        roc_vis_1155_zs_out, p_vis_1155_zs_out        =   compute_roc_auc(test_data['zscored data'][stim_vis_11_out, :], test_data['zscored data'][stim_vis_55_out, :])
        roc_match_1155_zs_out, p_match_1155_zs_out    =   compute_roc_auc(test_data['zscored data'][stim_match_1155_out, :], test_data['zscored data'][stim_nomatch_1155_out, :])

        roc_vis_1551_zs_out, p_vis_1551_zs_out        =   compute_roc_auc(test_data['zscored data'][stim_vis_15_out, :], test_data['zscored data'][stim_vis_51_out, :])
        roc_match_1551_zs_out, p_match_1551_zs_out    =   compute_roc_auc(test_data['zscored data'][stim_match_1551_out, :], test_data['zscored data'][stim_nomatch_1551_out, :])

        roc_match_zs_out, p_match_zs_out  =   compute_roc_auc(test_data['zscored data'][stim_match_out, :], test_data['zscored data'][stim_nomatch_out, :])

        # zs_in=np.mean(np.mean(test_data['zscored data'][test_data['sample pos']==1,:], axis=0), axis=1)
        # zs_out=np.mean(np.mean(test_data['zscored data'][test_data['sample pos']==-1,:], axis=0), axis=1)

        
        ## choice probability
        rts=test_data['reaction times'][stim_match_in]
        select_trials=stim_match_in[rts>0]
        rts=test_data['reaction times'][select_trials]
        acti=np.mean(test_data['zscored data'][select_trials,300+50:300+200], axis=1)
        cp_coeff_in, cp_p_in=stats.pearsonr(rts, acti)

        rts=test_data['reaction times'][stim_match_out]
        select_trials=stim_match_out[rts>0]
        rts=test_data['reaction times'][select_trials]
        acti=np.mean(test_data['zscored data'][select_trials,300+50:300+200], axis=1)
        cp_coeff_out, cp_p_out=stats.pearsonr(rts, acti)

        ########for archive########
        # # test o1 vs o5
        # vis_to1_in     =   np.sort(np.concatenate([s_in_11_t_11, s_in_15_t_11, s_in_51_t_11, s_in_55_t_11,
        #                                             s_in_11_t_15, s_in_15_t_15, s_in_51_t_15, s_in_55_t_15]))

        # vis_to5_in     =   np.sort(np.concatenate([s_in_11_t_51, s_in_15_t_51, s_in_51_t_51, s_in_55_t_51,
        #                                         s_in_11_t_55, s_in_15_t_55, s_in_51_t_55, s_in_55_t_55]))

        # vis_tc1_in     =   np.sort(np.concatenate([s_in_11_t_11, s_in_15_t_11, s_in_51_t_11, s_in_55_t_11,
        #                                         s_in_11_t_51, s_in_15_t_51, s_in_51_t_51, s_in_55_t_51]))

        # vis_tc5_in     =   np.sort(np.concatenate([s_in_11_t_15, s_in_15_t_15, s_in_51_t_15, s_in_55_t_15,
        #                                         s_in_11_t_55, s_in_15_t_55, s_in_51_t_55, s_in_55_t_55]))

        # match_trials_in    =   np.sort(np.concatenate([s_in_11_t_11, s_in_15_t_15, s_in_51_t_51, s_in_55_t_51]))
        # nomatch_trials_in  =   np.sort(np.concatenate([s_in_11_t_15, s_in_11_t_51,s_in_11_t_55,
        #                                             s_in_15_t_11, s_in_15_t_51,s_in_11_t_55,
        #                                             s_in_51_t_11, s_in_51_t_15,s_in_51_t_55,
        #                                             s_in_55_t_11, s_in_55_t_15,s_in_55_t_51]))
        # o_test_av_in, p_o_test_av_in  =   compute_roc_auc(test_data['averaged data'][vis_to1_in, :], test_data['averaged data'][vis_to5_in, :])
        # c_test_av_in, p_c_test_av_in  =   compute_roc_auc(test_data['averaged data'][vis_tc1_in, :], test_data['averaged data'][vis_tc5_in, :])
        # m_test_av_in, p_m_test_av_in  =   compute_roc_auc(test_data['averaged data'][match_trials_in, :], test_data['averaged data'][nomatch_trials_in, :])

        # o_test_zs_in, p_o_test_zs_in  =   compute_roc_auc(test_data['zscored data'][vis_to1_in, :], test_data['zscored data'][vis_to5_in, :])
        # c_test_zs_in, p_c_test_zs_in  =   compute_roc_auc(test_data['zscored data'][vis_tc1_in, :], test_data['zscored data'][vis_tc5_in, :])
        # m_test_zs_in, p_m_test_zs_in  =   compute_roc_auc(test_data['zscored data'][match_trials_in, :], test_data['zscored data'][nomatch_trials_in, :])

        return{'name': test_data['name'],
                'reaction times': test_data['reaction times'],      
                'roc v vis in zs 1155'  :   roc_vis_1155_zs_in,
                'roc v vis in zs 1551'  :   roc_vis_1551_zs_in,
                'roc v match in zs 1155':   roc_match_1155_zs_in,
                'roc v match in zs 1551':   roc_match_1551_zs_in,
                'p v vis in zs 1155'    :   p_vis_1155_zs_in,
                'p v vis in zs 1551'    :   p_vis_1551_zs_in,
                'p v match in zs 1155'  :   p_match_1155_zs_in,
                'p v match in zs 1551'  :   p_match_1551_zs_in,
                'roc v match in zs'     :   roc_match_zs_in,
                'p v match in zs'       :   p_match_zs_in,
                'roc v vis out zs 1155'  :   roc_vis_1155_zs_out,
                'roc v vis out zs 1551'  :   roc_vis_1551_zs_out,
                'roc v match out zs 1155':   roc_match_1155_zs_out,
                'roc v match out zs 1551':   roc_match_1551_zs_out,
                'p v vis out zs 1155'    :   p_vis_1155_zs_out,
                'p v vis out zs 1551'    :   p_vis_1551_zs_out,
                'p v match out zs 1155'  :   p_match_1155_zs_out,
                'p v match out zs 1551'  :   p_match_1551_zs_out,
                'roc v match out zs'     :   roc_match_zs_out,
                'p v match out zs'       :   p_match_zs_out,
                'cp coeff in'       :   cp_coeff_in,
                'cp p in'           :   cp_p_in,
                'cp coeff out'      :   cp_coeff_out,
                'cp p out'          :   cp_p_out}

def reorganize_data(ocm_result):
    name=[]
    rt=[]
    roc_1155_vis_in     =   []#np.empty([len(ocm_result), 800])
    roc_1551_vis_in     =   []#np.empty([len(ocm_result), 800])
    roc_1155_match_in   =   []#np.empty([len(ocm_result), 800])
    roc_1551_match_in   =   []#np.empty([len(ocm_result), 800])
    roc_match_in        =   []    

    p_1155_vis_in       =   []#np.empty([len(ocm_result), 800])
    p_1551_vis_in       =   []#np.empty([len(ocm_result), 800])
    p_1155_match_in     =   []#np.empty([len(ocm_result), 800])
    p_1551_match_in     =   []#np.empty([len(ocm_result), 800])
    p_match_in          =   []    

    roc_1155_vis_out    =   []#np.empty([len(ocm_result), 800])
    roc_1551_vis_out    =   []#np.empty([len(ocm_result), 800])
    roc_1155_match_out  =   []#np.empty([len(ocm_result), 800])
    roc_1551_match_out  =   []#np.empty([len(ocm_result), 800])
    roc_match_out       =   []    

    p_1155_vis_out      =   []#np.empty([len(ocm_result), 800])
    p_1551_vis_out      =   []#np.empty([len(ocm_result), 800])
    p_1155_match_out    =   []#np.empty([len(ocm_result), 800])
    p_1551_match_out    =   []#np.empty([len(ocm_result), 800])
    p_match_out         =   []    


    cp_coeff_in     =   []
    cp_coeff_out    =   []
    cp_p_in         =   []
    cp_p_out        =   []

    for i in range(len(ocm_result)):
        
        if ocm_result[i] is not None:  
            name.append(ocm_result[i]['name'])
            rt.append(ocm_result[i]['reaction times'])
            roc_1155_vis_in.append(ocm_result[i]['roc v vis in zs 1155'])
            roc_1155_match_in.append(ocm_result[i]['roc v match in zs 1155'])
            roc_1551_vis_in.append(ocm_result[i]['roc v vis in zs 1551'])
            roc_1551_match_in.append(ocm_result[i]['roc v match in zs 1551'])
            roc_match_in.append(ocm_result[i]['roc v match in zs'])

            p_1155_vis_in.append(ocm_result[i]['p v vis in zs 1155'])
            p_1155_match_in.append(ocm_result[i]['p v match in zs 1155'])
            p_1551_vis_in.append(ocm_result[i]['p v vis in zs 1551'])
            p_1551_match_in.append(ocm_result[i]['p v match in zs 1551'])
            p_match_in.append(ocm_result[i]['p v match in zs'])


            roc_1155_vis_out.append(ocm_result[i]['roc v vis out zs 1155'])
            roc_1155_match_out.append(ocm_result[i]['roc v match out zs 1155'])
            roc_1551_vis_out.append(ocm_result[i]['roc v vis out zs 1551'])
            roc_1551_match_out.append(ocm_result[i]['roc v match out zs 1551'])
            roc_match_out.append(ocm_result[i]['roc v match out zs'])

            p_1155_vis_out.append(ocm_result[i]['p v vis out zs 1155'])
            p_1155_match_out.append(ocm_result[i]['p v match out zs 1155'])
            p_1551_vis_out.append(ocm_result[i]['p v vis out zs 1551'])
            p_1551_match_out.append(ocm_result[i]['p v match out zs 1551'])
            p_match_out.append(ocm_result[i]['p v match out zs'])

            cp_coeff_in.append(ocm_result[i]['cp coeff in'])
            cp_coeff_out.append(ocm_result[i]['cp coeff out'])
            cp_p_in.append(ocm_result[i]['cp p in'])
            cp_p_out.append(ocm_result[i]['cp p out'])

    roc_1155_vis_in=np.concatenate(roc_1155_vis_in)
    roc_1155_vis_in=np.concatenate(roc_1155_vis_in)
    roc_1155_match_in=np.concatenate(roc_1155_match_in)
    roc_1155_match_in=np.concatenate(roc_1155_match_in)
    roc_match_in=np.concatenate(roc_match_in)

    p_1155_vis_in=np.concatenate(p_1155_vis_in)
    p_1155_vis_in=np.concatenate(p_1155_vis_in)
    p_1155_match_in=np.concatenate(p_1155_match_in)
    p_1155_match_in=np.concatenate(p_1155_match_in)
    p_match_in=np.concatenate(p_match_in)
    

    roc_1155_vis_out=np.concatenate(roc_1155_vis_out)
    roc_1155_vis_out=np.concatenate(roc_1155_vis_out)
    roc_1155_match_out=np.concatenate(roc_1155_match_out)
    roc_1155_match_out=np.concatenate(roc_1155_match_out)
    roc_match_out=np.concatenate(roc_match_out)

    p_1155_vis_out=np.concatenate(p_1155_vis_out)
    p_1155_vis_out=np.concatenate(p_1155_vis_out)
    p_1155_match_out=np.concatenate(p_1155_match_out)
    p_1155_match_out=np.concatenate(p_1155_match_out)
    p_match_out=np.concatenate(p_match_out)
    
    cp_coeff_in=np.concatenate(cp_coeff_in)
    cp_coeff_out=np.concatenate(cp_coeff_out)
    cp_p_in=np.concatenate(cp_p_in)
    cp_p_out=np.concatenate(cp_p_out)
    rt=np.concatenate(rt)

    return {'name':name,
            'reaction times': rt,
            'roc v vis in 1155'     :   roc_1155_vis_in,
            'roc v vis in 1551'     :   roc_1155_vis_in,
            'roc v match in 1155'   :   roc_1155_match_in,
            'roc v match in 1551'   :   roc_1155_match_in,
            'roc v match in'        :   roc_match_in,
            'p v vis in 1155'       :   p_1155_vis_in,
            'p v vis in 1551'       :   p_1155_vis_in,
            'p v match in 1155'     :   p_1155_match_in,
            'p v match in 1551'     :   p_1155_match_in,
            'p v match in'          :   p_match_in,
            'roc v vis out 1155'    :   roc_1155_vis_out,
            'roc v vis out 1551'    :   roc_1155_vis_out,
            'roc v match out 1155'  :   roc_1155_match_out,
            'roc v match out 1551'  :   roc_1155_match_out,
            'roc v match out'       :   roc_match_out,
            'p v vis out 1155'      :   p_1155_vis_out,
            'p v vis out 1551'      :   p_1155_vis_out,
            'p v match out 1155'    :   p_1155_match_out,
            'p v match out 1551'    :   p_1155_match_out,
            'p v match out'         :   p_match_out,
            'cp coeff in'           :   cp_coeff_in,
            'cp coeff out'          :   cp_coeff_out,
            'cp p in'               :   cp_p_in,
            'cp p out'              :   cp_p_out}

with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_test_zsc_w50_s1m300p500', 'rb') as handle:
    test_lip = pickle.load(handle)

numcells=len(test_lip)
# for i in tqdm(range(len(test_lip))):
    
#     tmp=ocm(test_lip[i])

ocm_result    =   Parallel(n_jobs = -1)(delayed(ocm)(cell) for cell in tqdm(test_lip[:numcells]))
lip_ocm_data    =reorganize_data(ocm_result)
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_roc_test_zsc_w50_s1m300p500', "wb") as fp: 
    pickle.dump(lip_ocm_data, fp)
test_lip=[]
lip_ocm_data=[]


with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_test_zsc_w50_s1m300p500', 'rb') as handle:
    test_pfc = pickle.load(handle)
numcells=len(test_pfc)
ocm_result    =   Parallel(n_jobs = -1)(delayed(ocm)(cell) for cell in tqdm(test_pfc[:numcells]))
pfc_ocm_data    =reorganize_data(ocm_result)
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_roc_test_zsc_w50_s1m300p500', "wb") as fp: 
    pickle.dump(pfc_ocm_data, fp)
test_pfc=[]
pfc_ocm_data=[]

with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_test_zsc_w50_s1m300p500', 'rb') as handle:
    test_v4 = pickle.load(handle)

numcells=len(test_v4)
ocm_result    =   Parallel(n_jobs = -1)(delayed(ocm)(cell) for cell in tqdm(test_v4[:numcells]))
v4_ocm_data    =reorganize_data(ocm_result)
with open('/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_roc_test_zsc_w50_s1m300p500', "wb") as fp: 
    pickle.dump(v4_ocm_data, fp)
    