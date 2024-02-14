
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


def extract_test_trials(neu_data, align_on, code, select_block, time_before_sample): 
    sp_in_on,mask_in  =   align_trials.align_on(
            sp_samples  =   neu_data.sp_samples,
            code_samples=   neu_data.code_samples,
            code_numbers=   neu_data.code_numbers,
            trial_error =   neu_data.trial_error,
            block       =   neu_data.block,
            pos_code    =   neu_data.pos_code,
            select_block=   select_block,
            select_pos  =   code,
            event       =   'test_on_' + str(align_on+1),
            time_before =   time_before_sample,
            error_type  =   0,
        )

    sample_id           =   neu_data.sample_id[mask_in]
    sample_orientation  =   (sample_id/10).astype(int)
    sample_color        =   (sample_id%10).astype(int)

    test_stim           =   neu_data.test_stimuli[mask_in]

    match_stim          =   np.where(sample_id==test_stim[:,align_on])[0]
    nomatch_stim        =   np.where(np.logical_and(sample_id!=test_stim[:,align_on], sample_id!=0))[0]

    so1_trials          =   nomatch_stim[np.where(sample_orientation[nomatch_stim]==1)[0]]
    so5_trials          =   nomatch_stim[np.where(sample_orientation[nomatch_stim]==5)[0]]
    sc1_trials          =   nomatch_stim[np.where(sample_color[nomatch_stim]==1)[0]]
    sc5_trials          =   nomatch_stim[np.where(sample_color[nomatch_stim]==5)[0]]

    so1_trials_test_id  =   test_stim[so1_trials, align_on].astype(int)
    so5_trials_test_id  =   test_stim[so5_trials, align_on].astype(int)
    sc1_trials_test_id  =   test_stim[sc1_trials, align_on].astype(int)
    sc5_trials_test_id  =   test_stim[sc5_trials, align_on].astype(int)

    sn_trials           =   np.where(sample_id==0)[0]
    sn_trials_test_id   =   test_stim[sn_trials, align_on].astype(int)
    return {'data'      :   sp_in_on[:,:800], 
            'sample id' :   sample_id,
            'test id'   :   test_stim[:,align_on],
            'so1_trials':   so1_trials,
            'so5_trials':   so5_trials,
            'sc1_trials':   sc1_trials,
            'sc5_trials':   sc5_trials,
            'sn_trials' :   sn_trials,
            'match_stim':   match_stim,
            'nomatch_stim': nomatch_stim, 
            'test_id_so1_trials':  so1_trials_test_id,
            'test_id_so5_trials':  so5_trials_test_id, 
            'test_id_sc1_trials':  sc1_trials_test_id, 
            'test_id_sc5_trials':  sc5_trials_test_id, 
            'test_id_sn_trials' :  sn_trials_test_id}

def dat_to_test(cell):
      neu_data    =   NeuronData.from_python_hdf5(cell)
      date_time   =   neu_data.date_time
      code          =     1
      select_block  =     1
      time_before_sample=200
      if any(neu_data.block==1):
            test_data=[]
            sample_id=[]
            test_id=[]
            match_trials=[]
            nomatch_trials=[]
            test_so1_trials=[]
            test_so5_trials=[]
            test_sc1_trials=[]
            test_sc5_trials=[] 
            test_sn_trials=[]
            test_id_so1_trials=[]
            test_id_so5_trials=[]
            test_id_sc1_trials=[]
            test_id_sc5_trials=[]
            test_id_sn_trials =[]
            for i in range(neu_data.test_stimuli.shape[1]):
                  locals()['test' + str(i)] = extract_test_trials(neu_data, align_on=i, code=code, select_block=select_block, time_before_sample=time_before_sample)
                  test_data.append(locals()['test' + str(i)]['data'])
                  sample_id.append(locals()['test' + str(i)]['sample id'])
                  test_id.append(locals()['test' + str(i)]['test id'])
                  test_id_so1_trials.append(locals()['test' + str(i)]['test_id_so1_trials'])
                  test_id_so5_trials.append(locals()['test' + str(i)]['test_id_so5_trials'])
                  test_id_sc1_trials.append(locals()['test' + str(i)]['test_id_sc1_trials'])
                  test_id_sc5_trials.append(locals()['test' + str(i)]['test_id_sc5_trials'])
                  test_id_sn_trials.append(locals()['test' + str(i)]['test_id_sn_trials'])
                  if i==0:
                        match_trials.append(locals()['test' + str(i)]['match_stim']) 
                        nomatch_trials.append(locals()['test' + str(i)]['nomatch_stim']) 
                        test_so1_trials.append(locals()['test' + str(i)]['so1_trials']) 
                        test_so5_trials.append(locals()['test' + str(i)]['so5_trials'])
                        test_sc1_trials.append(locals()['test' + str(i)]['sc1_trials'])
                        test_sc5_trials.append(locals()['test' + str(i)]['sc5_trials'])
                        test_sn_trials.append(locals()['test' + str(i)]['sn_trials'])
                  else:
                        match_trials.append(locals()['test' + str(i)]['match_stim']+len(locals()['test' + str(i-1)]['data'])) 
                        nomatch_trials.append(locals()['test' + str(i)]['nomatch_stim']+len(locals()['test' + str(i-1)]['data'])) 
                        test_so1_trials.append(locals()['test' + str(i)]['so1_trials']+len(locals()['test' + str(i-1)]['data'])) 
                        test_so5_trials.append(locals()['test' + str(i)]['so5_trials']+len(locals()['test' + str(i-1)]['data'])) 
                        test_sc1_trials.append(locals()['test' + str(i)]['sc1_trials']+len(locals()['test' + str(i-1)]['data'])) 
                        test_sn_trials.append(locals()['test' + str(i)]['sn_trials']+len(locals()['test' + str(i-1)]['data'])) 
            
            
            test={'name':date_time[:10] + '_'+ neu_data.cluster_group + '_'+ str(neu_data.cluster_number),
                  'data'            :   np.concatenate(test_data),
                  'sample id'       :   np.concatenate(sample_id),
                  'test id'         :   np.concatenate(test_id),
                  'match trials'    :   np.concatenate(match_trials),
                  'nomatch_trials'  :   np.concatenate(nomatch_trials),
                  'so1 trials'      :   np.concatenate(test_so1_trials),
                  'so5_trials'      :   np.concatenate(test_so5_trials),
                  'sc1_trials'      :   np.concatenate(test_sc1_trials),
                  'sc5_trials'      :   np.concatenate(test_sc5_trials),
                  'sn_trials'       :   np.concatenate(test_sn_trials),
                  'test_id_so1_trials'  : np.concatenate(test_id_so1_trials),
                  'test_id_so5_trials'  :np.concatenate(test_id_so5_trials),
                  'test_id_sc1_trials'  :np.concatenate(test_id_sc1_trials),
                  'test_id_sc5_trials'  :np.concatenate(test_id_sc5_trials),
                  'test_id_sn_trials'   :np.concatenate(test_id_sn_trials)}
            return test

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


numcells=len(neurons_lip_files)
lip_test    =   Parallel(n_jobs = -1)(delayed(dat_to_test)(cell) for cell in tqdm(neurons_lip_files[:numcells]))

with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/LIP_test", "wb") as fp: 
    pickle.dump(lip_test, fp)


numcells=len(neurons_pfc_files)
pfc_test    =   Parallel(n_jobs = -1)(delayed(dat_to_test)(cell) for cell in tqdm(neurons_pfc_files[:numcells]))
with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/PFC_test", "wb") as fp: 
    pickle.dump(pfc_test, fp)


numcells=len(neurons_v4_files)
v4_test    =   Parallel(n_jobs = -1)(delayed(dat_to_test)(cell) for cell in tqdm(neurons_v4_files[:numcells]))
with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/v4_test", "wb") as fp: 
    pickle.dump(v4_test, fp)


# test_o1_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==1)]
# test_o2_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==2)]
# test_o3_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==3)]
# test_o4_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==4)]
# test_o5_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==5)]
# test_o6_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==6)]
# test_o7_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==7)]
# test_o8_so1_trials  =   test_so1_trials[np.where((test_id_so1_trials/10).astype(int)==8)]

# test_o1_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==1)]
# test_o2_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==2)]
# test_o3_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==3)]
# test_o4_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==4)]
# test_o5_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==5)]
# test_o6_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==6)]
# test_o7_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==7)]
# test_o8_so5_trials  =   test_so5_trials[np.where((test_id_so5_trials/10).astype(int)==8)]

# test_c1_sc1_trials  =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==1)]
# test_c2_sc1_trials  =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==2)]
# test_c3_sc1_trials  =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==3)]
# test_c4_sc1_trials  =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==4)]
# test_n_sc1_trials   =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==5)]
# test_c6_sc1_trials  =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==6)]
# test_c7_sc1_trials  =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==7)]
# test_c8_sc1_trials  =   test_sc1_trials[np.where((test_id_sc1_trials%10).astype(int)==8)]

# test_c1_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==1)]
# test_c2_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==2)]
# test_c3_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==3)]
# test_c4_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==4)]
# test_c5_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==5)]
# test_c6_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==6)]
# test_c7_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==7)]
# test_c8_sc5_trials  =   test_sc5_trials[np.where((test_id_sc5_trials%10).astype(int)==8)]

# test_o1_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==1)]
# test_o2_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==2)]
# test_o3_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==3)]
# test_o4_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==4)]
# test_o5_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==5)]
# test_o6_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==6)]
# test_o7_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==7)]
# test_o8_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials/10).astype(int)==8)]

# test_c1_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==1)]
# test_c2_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==2)]
# test_c3_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==3)]
# test_c4_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==4)]
# test_c5_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==5)]
# test_c6_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==6)]
# test_c7_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==7)]
# test_c8_sn_trials   =   test_sn_trials[np.where((test_id_sn_trials%10).astype(int)==8)]


# plt.polar(np.linspace(0,math.pi*2,9)[:-1],[np.mean(test_data[test_o1_so1_trials,250:450]*1000),
#           np.mean(test_data[test_o2_so1_trials,250:450]*1000),
#           np.mean(test_data[test_o3_so1_trials,250:450]*1000),
#           np.mean(test_data[test_o4_so1_trials,250:450]*1000),
#           np.mean(test_data[test_o5_so1_trials,250:450]*1000),
#           np.mean(test_data[test_o6_so1_trials,250:450]*1000),
#           np.mean(test_data[test_o7_so1_trials,250:450]*1000),
#           np.mean(test_data[test_o8_so1_trials,250:450]*1000)], 'o')

# plt.polar(np.linspace(0,math.pi*2,9)[:-1],[np.mean(test_data[test_o1_so5_trials,250:450]*1000),
#           np.mean(test_data[test_o2_so5_trials,250:450]*1000),
#           np.mean(test_data[test_o3_so5_trials,250:450]*1000),
#           np.mean(test_data[test_o4_so5_trials,250:450]*1000),
#           np.mean(test_data[test_o5_so5_trials,250:450]*1000),
#           np.mean(test_data[test_o6_so5_trials,250:450]*1000),
#           np.mean(test_data[test_o7_so5_trials,250:450]*1000),
#           np.mean(test_data[test_o8_so5_trials,250:450]*1000)], 'o')

# plt.polar(np.linspace(0,math.pi*2,9)[:-1],[np.mean(test_data[test_o1_sn_trials,250:450]*1000),
#           np.mean(test_data[test_o2_sn_trials,250:450]*1000),
#           np.mean(test_data[test_o3_sn_trials,250:450]*1000),
#           np.mean(test_data[test_o4_sn_trials,250:450]*1000),
#           np.mean(test_data[test_o5_sn_trials,250:450]*1000),
#           np.mean(test_data[test_o6_sn_trials,250:450]*1000),
#           np.mean(test_data[test_o7_sn_trials,250:450]*1000),
#           np.mean(test_data[test_o8_sn_trials,250:450]*1000)], 'o')
# plt.legend(['o1 trials', 'o5 trials', 'neutral trials'])



# plt.polar(np.linspace(0,math.pi*2,9)[:-1],[np.mean(test_data[test_c1_sc1_trials,250:450]*1000),
#           np.mean(test_data[test_c2_sc1_trials,250:450]*1000),
#           np.mean(test_data[test_c3_sc1_trials,250:450]*1000),
#           np.mean(test_data[test_c4_sc1_trials,250:450]*1000),
#           np.mean(test_data[test_c5_sc1_trials,250:450]*1000),
#           np.mean(test_data[test_c6_sc1_trials,250:450]*1000),
#           np.mean(test_data[test_c7_sc1_trials,250:450]*1000),
#           np.mean(test_data[test_c8_sc1_trials,250:450]*1000)], 'o')

# plt.polar(np.linspace(0,math.pi*2,9)[:-1],[np.mean(test_data[test_c1_sc5_trials,250:450]*1000),
#           np.mean(test_data[test_c2_sc5_trials,250:450]*1000),
#           np.mean(test_data[test_c3_sc5_trials,250:450]*1000),
#           np.mean(test_data[test_c4_sc5_trials,250:450]*1000),
#           np.mean(test_data[test_c5_sc5_trials,250:450]*1000),
#           np.mean(test_data[test_c6_sc5_trials,250:450]*1000),
#           np.mean(test_data[test_c7_sc5_trials,250:450]*1000),
#           np.mean(test_data[test_c8_sc5_trials,250:450]*1000)], 'o')

# plt.polar(np.linspace(0,math.pi*2,9)[:-1],[np.mean(test_data[test_c1_sn_trials,250:450]*1000),
#           np.mean(test_data[test_c2_sn_trials,250:450]*1000),
#           np.mean(test_data[test_c3_sn_trials,250:450]*1000),
#           np.mean(test_data[test_c4_sn_trials,250:450]*1000),
#           np.mean(test_data[test_c5_sn_trials,250:450]*1000),
#           np.mean(test_data[test_c6_sn_trials,250:450]*1000),
#           np.mean(test_data[test_c7_sn_trials,250:450]*1000),
#           np.mean(test_data[test_c8_sn_trials,250:450]*1000)], 'o')
# plt.legend(['c1 trials', 'c5 trials', 'neutral trials'])



# win=100
# step=1
# test_data_avg_sp, test_data_z_sp  =   moving_average(data=test_data,win=win, step=step)




# plt.plot(np.mean(test_data_avg_sp[test_o1_so1_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o2_so1_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o3_so1_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o4_so1_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o5_so1_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o6_so1_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o7_so1_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o8_so1_trials,:], axis=0)*1000)


# plt.plot(np.mean(test_data_avg_sp[test_o1_so5_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o2_so5_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o3_so5_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o4_so5_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o5_so5_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o6_so5_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o7_so5_trials,:], axis=0)*1000)
# plt.plot(np.mean(test_data_avg_sp[test_o8_so5_trials,:], axis=0)*1000)


# so1c1_to1c1     =   np.where(np.logical_and(test_id==11, sample_id==11))[0]
# so1c5_to1c5     =   np.where(np.logical_and(test_id==15, sample_id==15))[0]
# so5c1_to5c1     =   np.where(np.logical_and(test_id==51, sample_id==51))[0]
# so5c5_to5c5     =   np.where(np.logical_and(test_id==55, sample_id==55))[0]

# sother_to1c1     =   np.where(np.logical_and(test_id==11, sample_id!=11, sample_id!=0))[0]
# sother_to1c5     =   np.where(np.logical_and(test_id==15, sample_id!=15, sample_id!=0))[0]
# sother_to5c1     =   np.where(np.logical_and(test_id==51, sample_id!=51, sample_id!=0))[0]
# sother_to5c5     =   np.where(np.logical_and(test_id==55, sample_id!=55, sample_id!=0))[0]




# plt.plot(np.mean(test_data_avg_sp[so1c1_to1c1,:], axis=0)*1000, color='c')
# plt.plot(np.mean(test_data_avg_sp[so1c5_to1c5,:], axis=0)*1000, color='m')
# plt.plot(np.mean(test_data_avg_sp[so5c1_to5c1,:], axis=0)*1000, color='y')
# plt.plot(np.mean(test_data_avg_sp[so5c5_to5c5,:], axis=0)*1000, color='k')

# plt.plot(np.mean(test_data_avg_sp[sother_to1c1,:], axis=0)*1000, color='c', linestyle='dashed')
# plt.plot(np.mean(test_data_avg_sp[sother_to1c5,:], axis=0)*1000, color='m', linestyle='dashed')
# plt.plot(np.mean(test_data_avg_sp[sother_to5c1,:], axis=0)*1000, color='y', linestyle='dashed')
# plt.plot(np.mean(test_data_avg_sp[sother_to5c5,:], axis=0)*1000, color='k', linestyle='dashed')



