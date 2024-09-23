## Activation index
### 1st: Compute activation index using: Users/losadac/data_stats/rf_position/activation_index.py
This script saves (in /envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/):
                    -   activation_idx_{area}.png
                    -   population_selectivity_{area}.h5
                    -   population_selectivity_{area}.pkl
                    -   position_selectivity_{area}.png
                    -   {area}_plots/{datetime}_{subject}_{...}.png

### 2nd: Manual verification of the neurons classified as ipsilateral (inside folder /envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/{area}_plots/)
The ones that present a strong ipsilateral response were moved to /envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/{area}_ipsilateral/

### 3rd: Compute a df indicating whether the cell response is ipsi or contra lateral to the position of the stimulus.
(Using /Users/losadac/data_stats/rf_position/rf_position_df.ipynb)
This script saves the df in /envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_{area}.csv

## Selectivity
### 1st: Compute neurons selectivity with: /Users/losadac/selectivity/__ main__.py
The scrip saves a file called 'population_selectivity.pkl' in 
/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/
### 2nd: Compute plots using /Users/losadac/selectivity/population_selectivity.ipynb
This notebook save plots and:
* a file called: '{area}_no_neutral_inout_selectivity.csv' that contains neurons presenting no selectivity to neutral
sample inside the rf vs outside
* a file called: '{area}_neutral_inout_selectivity.csv' that contains neurons presenting selectivity to neutral
sample inside the rf vs outside
