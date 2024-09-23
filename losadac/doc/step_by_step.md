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

