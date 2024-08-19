
import numpy as np
import numpy.matlib as mt
import h5py
import math 
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import align_trials
from ephysvibe.task import task_constants

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

def extract_test_trials(neu_data, align_on, select_block, time_before_sample): 
    
    #IN
    sp_in_on,mask_in  =   align_trials.align_on(
            sp_samples  =   neu_data.sp_samples,
            code_samples=   neu_data.code_samples,
            code_numbers=   neu_data.code_numbers,
            trial_error =   neu_data.trial_error,
            block       =   neu_data.block,
            pos_code    =   neu_data.pos_code,
            select_block=   select_block,
            select_pos  =   1,
            event       =   'test_on_' + str(align_on+1),
            time_before =   time_before_sample,
            error_type  =   0,
        )
    
    sp_out_on,mask_out  =   align_trials.align_on(
            sp_samples  =   neu_data.sp_samples,
            code_samples=   neu_data.code_samples,
            code_numbers=   neu_data.code_numbers,
            trial_error =   neu_data.trial_error,
            block       =   neu_data.block,
            pos_code    =   neu_data.pos_code,
            select_block=   select_block,
            select_pos  =   -1,
            event       =   'test_on_' + str(align_on+1),
            time_before =   time_before_sample,
            error_type  =   0,
        )
    
    sample_in_id    =   neu_data.sample_id[mask_in]
    sample_out_id   =   neu_data.sample_id[mask_out]
    sample_in_pos   =   neu_data.pos_code[mask_in]
    sample_out_pos  =   neu_data.pos_code[mask_out]
    test_in_stim    =   neu_data.test_stimuli[mask_in, align_on]
    test_out_stim   =   neu_data.test_stimuli[mask_out, align_on]
    dist_in_stim    =   neu_data.test_distractor[mask_in, align_on]
    dist_out_stim   =   neu_data.test_distractor[mask_out, align_on]

    rt_in  =   np.empty(np.sum(mask_in))*np.nan
    rt_out =   np.empty(np.sum(mask_out))*np.nan

    match   =   np.where(mask_in)[0][np.where(neu_data.sample_id[mask_in]==neu_data.test_stimuli[mask_in, align_on])]
    rt_in[np.where(neu_data.sample_id[mask_in]==neu_data.test_stimuli[mask_in, align_on])]=align_trials.get_rt(neu_data.code_numbers[match], neu_data.code_samples[match])

    match   =   np.where(mask_out)[0][np.where(neu_data.sample_id[mask_out]==neu_data.test_stimuli[mask_out, align_on])]
    rt_out[np.where(neu_data.sample_id[mask_out]==neu_data.test_stimuli[mask_out, align_on])]=align_trials.get_rt(neu_data.code_numbers[match], neu_data.code_samples[match])

    data        =   np.concatenate([sp_in_on, sp_out_on])
    sample_id   =   np.concatenate([sample_in_id, sample_out_id])
    sample_pos  =   np.concatenate([sample_in_pos, sample_out_pos])
    test_stim   =   np.concatenate([test_in_stim, test_out_stim])
    dist_stim   =   np.concatenate([dist_in_stim, dist_out_stim])
    rt          =   np.concatenate([rt_in, rt_out])
    return {'raw data':data[:,:800],
            'sample id': sample_id,
            'sample pos': sample_pos,
            'test id': test_stim,
            'distractor id': dist_stim,
            'reaction times': rt}

def dat_to_test(cell):
    neu_data    =   NeuronData.from_python_hdf5(cell)
    date_time   =   neu_data.date_time
    select_block  =     1
    time_before_sample= 300
    win=100
    step=1
    # tmp =   np.logical_and(np.logical_and(neu_data.block==1, neu_data.trial_error==0), neu_data.pos_code==1)
    # rtin=align_trials.get_rt(neu_data.code_numbers[tmp], neu_data.code_samples[tmp])
    # tmp =   np.logical_and(np.logical_and(neu_data.block==1, neu_data.trial_error==0), neu_data.pos_code==-1)
    # rtout=align_trials.get_rt(neu_data.code_numbers[tmp], neu_data.code_samples[tmp])
    # rt=np.concatenate([rtin, rtout])
    if np.sum(neu_data.block==1)>100:
        data        =   []
        sample_id   =   []
        sample_pos  =   []
        test_id     =   []
        ditr_id     =   []
        test_num    =   []
        rt=[]
        
        for i in range(neu_data.test_stimuli.shape[1]):
            locals()['test' + str(i)] = extract_test_trials(neu_data, align_on=i, 
                                                            select_block=select_block,
                                                            time_before_sample=time_before_sample)
            # if locals()['test' + str(i)] ['sample id']==locals()['test' + str(i)] ['test_id']
            data.append(locals()['test' + str(i)] ['raw data'])
            sample_id.append(locals()['test' + str(i)] ['sample id'])
            sample_pos.append(locals()['test' + str(i)] ['sample pos'])
            test_id.append(locals()['test' + str(i)] ['test id'])
            ditr_id.append(locals()['test' + str(i)] ['distractor id'])
            test_num.append(np.ones(locals()['test' + str(i)] ['raw data'].shape[0])*i)
            rt.append(locals()['test' + str(i)] ['reaction times'])
        
        data        =   np.concatenate(data)        
        sample_id   =   np.concatenate(sample_id)  
        sample_pos  =   np.concatenate(sample_pos)  
        test_id     =   np.concatenate(test_id)  
        ditr_id     =   np.concatenate(ditr_id)  
        test_num    =   np.concatenate(test_num)  
        rt          =   np.concatenate(rt)
        
        data_avg, data_z    =   moving_average(data=data,win=win, step=step)

        return {'name': date_time[:10] + '_'+ neu_data.cluster_group + '_'+ str(neu_data.cluster_number),
                'raw data': data, 
                'sample id': sample_id, 
                'sample pos': sample_pos,
                'test id': test_id,
                'ditr id': ditr_id,
                'test num': test_num,
                'reaction times': rt},{'name': date_time[:10] + '_'+ neu_data.cluster_group + '_'+ str(neu_data.cluster_number),
                                   'averaged data': data_avg,
                                   'sample id': sample_id, 
                                    'sample pos': sample_pos,
                                    'test id': test_id,
                                    'ditr id': ditr_id,
                                    'test num': test_num,
                                    'reaction times': rt}, {'name': date_time[:10] + '_'+ neu_data.cluster_group + '_'+ str(neu_data.cluster_number),
                                                                'zscored data': data_z,
                                                                'sample id': sample_id, 
                                                                'sample pos': sample_pos,
                                                                'test id': test_id,
                                                                'ditr id': ditr_id,
                                                                'test num': test_num,
                                                                'reaction times': rt} 
                                                            

directory_b1    =   "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/"
bhv_directory   =   os.path.normpath(str(directory_b1) +  "/bhv/")

area="pfc"
neurons_pfc_directory =   os.path.normpath(str(directory_b1) + area + "/neurons/*.h5")
neurons_pfc_files     =   glob.glob(neurons_pfc_directory, recursive=True)

area="v4"
neurons_v4_directory =   os.path.normpath(str(directory_b1) + area + "/neurons/*.h5")
neurons_v4_files     =   glob.glob(neurons_v4_directory, recursive=True)

area="lip"
neurons_lip_directory =   os.path.normpath(str(directory_b1) + area + "/neurons/*.h5")
neurons_lip_files     =   glob.glob(neurons_lip_directory, recursive=True)

lip_go  =   True
pfc_go  =   True
v4_go   =   True

if lip_go==True:
    numcells=len(neurons_lip_files)
    
    lip_test    =   Parallel(n_jobs = -1)(delayed(dat_to_test)(cell) for cell in tqdm(neurons_lip_files[:numcells]))
    raw =   []
    avg =   []
    zsc =   []
    cond=   []
    for i in range(len(lip_test)):
         if lip_test[i] is not None:
            raw.append(lip_test[i][0])
            avg.append(lip_test[i][1])
            zsc.append(lip_test[i][2])
            # cond.append(lip_test[i][3])
    print('saving lip raw')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_test_raw_w100_s1m300p500", "wb") as fp: 
        pickle.dump(raw, fp)
    raw=[]
    print('saving lip avg')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_test_avg_w100_s1m300p500", "wb") as fp: 
        pickle.dump(avg, fp)
    avg=[]
    print('saving lip zscored')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_test_zsc_w100_s1m300p500", "wb") as fp: 
        pickle.dump(zsc, fp)
    zsc=[]
    # print('saving lip cond')
    # with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/lip_test_cond_w50_s1m300p500", "wb") as fp: 
    #     pickle.dump(cond, fp)
    # cond=[]
    lip_test=[]


if pfc_go==True:
    numcells=len(neurons_pfc_files)
    pfc_test    =   Parallel(n_jobs = -1)(delayed(dat_to_test)(cell) for cell in tqdm(neurons_pfc_files[:numcells]))
    raw =   []
    avg =   []
    zsc =   []
    cond=   []
    for i in range(len(pfc_test)):
         if pfc_test[i] is not None:
            raw.append(pfc_test[i][0])
            avg.append(pfc_test[i][1])
            zsc.append(pfc_test[i][2])
            # cond.append(pfc_test[i][3])
    
    print('saving pfc raw')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_test_raw_w100_s1m300p500", "wb") as fp: 
        pickle.dump(raw, fp)
    raw=[]
    print('saving pfc avg')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_test_avg_w100_s1m300p500", "wb") as fp: 
        pickle.dump(avg, fp)
    avg=[]
    print('saving pfc zscored')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_test_zsc_w100_s1m300p500", "wb") as fp: 
        pickle.dump(zsc, fp)
    zsc=[]
    # print('saving pfc cond')
    # with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/pfc_test_cond_w50_s1m300p500", "wb") as fp: 
    #     pickle.dump(cond, fp)
    cond=[]

if v4_go==True:

    numcells=len(neurons_v4_files)
    v4_test    =   Parallel(n_jobs = -1)(delayed(dat_to_test)(cell) for cell in tqdm(neurons_v4_files[:numcells]))
    raw =   []
    avg =   []
    zsc =   []
    cond=   []
    for i in range(len(v4_test)):
        if v4_test[i] is not None:
            raw.append(v4_test[i][0])
            avg.append(v4_test[i][1])
            zsc.append(v4_test[i][2])
            # cond.append(v4_test[i][3])
    
    print('saving v4 raw')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_test_raw_100_s1m300p500", "wb") as fp: 
        pickle.dump(raw, fp)
    raw=[]
    print('saving v4 avg')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_test_avg_100_s1m300p500", "wb") as fp: 
        pickle.dump(avg, fp)
    avg=[]
    print('saving v4 zscored')
    with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_test_zsc_100_s1m300p500", "wb") as fp: 
        pickle.dump(zsc, fp)
    zsc=[]
    # print('saving v4 cond')
    # with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/test/v4_test_cond_w100_s5m300p500", "wb") as fp: 
    #     pickle.dump(cond, fp)
    cond=[]

