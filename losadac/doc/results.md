# Results

## Selectivity
Results using non zero and inside 1.5 iqr trials
### All neurons
V4

![selectivity all neurons non zero percentile trials V4](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/v4_selectivity.jpg)

LIP

![selectivity all neurons non zero percentile trials LIP](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/lip_selectivity.jpg)

PFC

![selectivity all neurons non zero percentile trials PFC](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/pfc_selectivity.jpg)


* **Neutral status**:
In both cases (IN, OUT) the three areas present selectivity to the neutral status
    * In the IN condition some V4 neurons change selectivity during the delay. In the OUT there are very few neurons presenting selectivity and is mostly stable 
    * Most of LIP neurons are stable during the delay. Same in OUT (with less neurons)
    * Many PFC neurons change selectivity (stronger than in V4) 
* **Color and Orientation**:
    * Inside the RF all areas present neurons selective to color and orientation. As in LIP and PFC, V4 have neurons that become selective to color or orientation during the delay
    * In the Out condition very few neurons in V4 present selectivity. In LIP there are neurons selective to orientation but very few to color and in PFC we do find neurons selective to color and orientation.

#### Latencies
![all neurons non zero percentile trials latencies](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/latencies.jpg)

For color and orientation IN latencies are smaller in V4 followed by LIP and PFC. But for neutral PFC is first, folowed by V4 and LIP

In the OUT condition, the order for neutral is PFC, LIP, V4. The notable diference between LIP and V4 in this case is expected because V4 is mostly sensory and the stimuli is placed outside the RF.
Due to the low amount of neurons that are selective to color in the areas and orientation in V4, is only possible to compare the latencies in orientation between PFC and LIP. LIP has lower latency than PFC

### Neurons without selectivity to neutral in vs neutral out
Here we remove neurons showing selectivity to neutral IN vs neutral OUT to discart the visual and keep the cognitive response to the neutral stimulus.

V4

![selectivity no selective nIN nOUT neurons non zero percentile trials V4](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/v4_selectivity_no_selective_neutral_inout.jpg)

LIP

![selectivity no selective nIN nOUT neurons non zero percentile trials LIP](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/lip_selectivity_no_selective_neutral_inout.jpg)

PFC

![selectivity no selective nIN nOUT neurons non zero percentile trials PFC](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/pfc_selectivity_no_selective_neutral_inout.jpg)

 
#### Latencies

![all neurons non zero percentile trials latencies](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/percentile_with_nonzero/2024_09_23_17_09_29/latencies_no_selective_neutral_inout.jpg)

The order of the latencies remains the same as when using all the neurons. There is an increase in the difference between LIP and PFC for neutral (IN).

### Neurons with selectivity to neutral in vs neutral out
* V4

* LIP

* PFC

#### Latencies

## Euclidean distance between neutral and non neutral in the population space
Results using non zero and inside 1.5 iqr trials
### All neurons
IN 
![latencies all neurons IN](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_distance/euclidean/percentile_with_nonzero/all_neurons/2024_09_24_21_18_55/in_rf/distance.png)

OUT 
![latencies all neurons OUT](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_distance/euclidean/percentile_with_nonzero/all_neurons/2024_09_24_21_18_55/out_rf/distance.png)

Here we measure the distance between neural population representations of neutral vs non neutral stimuli. 
Latencies are the same in the three areas in the IN case but for OUT the order is PFC, LIP and V4. 
The magnitude is by far PFC the largest, followed by LIP and V4. 

### Neurons without selectivity to neutral in vs neutral out
IN 
![latencies neurons without selectivity to neutral in vs neutral out IN](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_distance/euclidean/percentile_with_nonzero/no_neutral_inout_selectivity/2024_09_25_09_50_13/in_rf/distance.png)

OUT 
![latencies neurons without selectivity to neutral in vs neutral out OUT](../../../../../../../../envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_distance/euclidean/percentile_with_nonzero/no_neutral_inout_selectivity/2024_09_25_09_50_13/out_rf/distance.png)


### Conclusion/ideas

* Changes in selectivity in V4 during the delay could be produced by top-down modulations comming from PFC where these changes are stronger.  
* Selectivity of V4 neurons during the delay, toghether with the decoding sample id results using the full population -> distributed wm (attentional / preparatory modulations ???)
* The order of the latencies show visual information (color and orientation) is encoded first in V4, LIP and last PFC. But the neutral information is first encoded by PFC. This could be an engagement signal comming from PFC to V4, gating the information to LIP.
* The idea of top-down modulations gating the information to LIP is based on tuning shifts towards the relevant stimuli *Ibos and Freedman [2014](https://www.cell.com/neuron/fulltext/S0896-6273(14)00695-3),[2016](https://www.cell.com/neuron/fulltext/S0896-6273(16)30410-X)* and the stable state observed when decoding from this area neutral vs non neutral.
