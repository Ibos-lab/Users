
import numpy as np
import numpy.matlib as mt

from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import align_trials
from ephysvibe.task import task_constants


import os 
import glob
import pickle

from joblib import Parallel, delayed
from tqdm import tqdm
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

def create_matrix_per_area(cell):
    win             =   100
    step            =   10
    
    select_block    =   1
    code            =   1
    
    time_before_sample  =   200
    timetotal_sample    =   time_before_sample+450*3
    
    time_before_t1  =   450
    timetotal_t1    =   time_before_t1+450*2
  
    neu_data    =   NeuronData.from_python_hdf5(cell)
    if any(neu_data.block==1):
        date_time   =   neu_data.date_time
        sp_sample_in_on,mask_sample_in = align_trials.align_on(
                sp_samples=neu_data.sp_samples,
                code_samples=neu_data.code_samples,
                code_numbers=neu_data.code_numbers,
                trial_error=neu_data.trial_error,
                block=neu_data.block,
                pos_code=neu_data.pos_code,
                select_block= select_block,
                select_pos= code,
                event ="sample_on",
                time_before = time_before_sample,
                error_type= 0,
            )
        
        sp_test_in_on,mask_test_in = align_trials.align_on(
                sp_samples=neu_data.sp_samples,
                code_samples=neu_data.code_samples,
                code_numbers=neu_data.code_numbers,
                trial_error=neu_data.trial_error,
                block=neu_data.block,
                pos_code=neu_data.pos_code,
                select_block= select_block,
                select_pos= code,
                event ="test_on_1",
                time_before = time_before_t1,
                error_type= 0,
            )
        sample_in_avg_sp, sample_in_std_sp  =   moving_average(data=sp_sample_in_on[:, :timetotal_sample],win=win, step=step)
        test1_in_avg_sp, test1_in_std_sp    =   moving_average(data=sp_test_in_on[:, :timetotal_t1],win=win, step=step)
        sample_in_avg_sp    =   sample_in_avg_sp[:,:-int(win/step)]
        sample_in_std_sp    =   sample_in_std_sp[:,:-int(win/step)]
        test1_in_avg_sp     =   test1_in_avg_sp[:,:-int(win/step)]
        test1_in_std_sp     =   test1_in_std_sp[:,:-int(win/step)]

        sample_id           =   neu_data.sample_id[mask_sample_in]

        return {'name': date_time[:10] + '_'+ neu_data.cluster_group + '_'+ str(neu_data.cluster_number),
            'Sample averaged'       :   sample_in_avg_sp,
            'Sample zscored'            :   sample_in_std_sp,
            'Test1 averaged'        :   test1_in_avg_sp,
            'Test1 zscored'             :   test1_in_std_sp,
            'Sample Id'             :   sample_id,
            'time_before_sample'    :   time_before_sample,
            'timetotal_sample'      :   timetotal_sample,
            'win'                   :   win,
            'step'                  :   step}
    # return {'name': date_time[:10] + '_'+ neu_data.cluster_group + '_'+ str(neu_data.cluster_number),
    #     'Sample averaged'       :   sample_in_avg_sp,
    #     'Sample zscored'            :   sample_in_std_sp,
    #     'Test1 averaged'        :   test1_in_avg_sp,
    #     'Test1 zscored'             :   test1_in_std_sp,
    #     'Sample Id'             :   sample_id,
    #     'time_before_sample'    :   time_before_sample,
    #     'timetotal_sample'      :   timetotal_sample,
    #     'win'                   :   win,
    #     'step'                  :   step}

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
lip_sample_in_avg_sp     =   Parallel(n_jobs = 6)(delayed(create_matrix_per_area)(cell) for cell in tqdm(neurons_lip_files[:numcells]))
lip_data=[]
for i in range(len(lip_sample_in_avg_sp)):
    if lip_sample_in_avg_sp[i]['Sample averaged'].shape[0]>1:
        lip_data.append(lip_sample_in_avg_sp)
with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/LIPavacti", "wb") as fp: 
    pickle.dump(lip_data, fp)

# numcells=len(neurons_v4_files)
# v4_sample_in_avg_sp     =   Parallel(n_jobs = -1)(delayed(create_matrix_per_area)(cell) for cell in tqdm(neurons_v4_files[:numcells]))
# v4_data=[]
# for i in range(len(v4_sample_in_avg_sp)):
#     if v4_sample_in_avg_sp[i]['Sample averaged'].shape[0]>1:
#         v4_data.append(v4_sample_in_avg_sp)

# with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/v4avacti", "wb") as fp: 
#     pickle.dump(v4_data, fp)


# numcells=len(neurons_pfc_files)
# pfc_sample_in_avg_sp     =   Parallel(n_jobs = -1)(delayed(create_matrix_per_area)(cell) for cell in tqdm(neurons_pfc_files[:numcells]))
# pfc_data=[]
# for i in range(len(pfc_sample_in_avg_sp)):
#     if pfc_sample_in_avg_sp[i]['Sample averaged'].shape[0]>1:
#         pfc_data.append(pfc_sample_in_avg_sp)
# with open("/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/averaged_structures/pfcavacti", "wb") as fp: 
#     pickle.dump(pfc_data, fp)        
