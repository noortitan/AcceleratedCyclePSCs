# Accelerated cycle aging test analysis for perovskite solar cells (AcceleratedCyclePSCs)

Version: 2023-11-29

This documentation is prepared as the workflow to accompany the following study:

**"How to accelerate outdoor ageing of perovskite solar cells by indoor testing"**

*(INSERT JOURNAL INFO)*

Noor Titan Putri Hartono (1), Artem Musiienko (1), Ulas Erdil (1), Zahra Loghman Nia (1), Mark Khenkin (1), Hans Köbler (1), Johannes Beckedahl (1), Florian Ruske (1), Rutger Schlatmann (1), Carolin Ulbrich (1), Antonio Abate (1)

Affiliations:

1. Helmholtz-Zentrum-Berlin für Materialien und Energie, 14109 Berlin, Germany

## Installation and Requirements

To install, just clone this repository:

`$ git clone https://github.com/noortitan/AccCyc.git`

`$ cd AccCyc`

To install the required package, create a virtual environment using Anaconda/ Miniconda (https://www.anaconda.com/download or https://docs.conda.io/en/latest/miniconda.html). The optional but recommended setup on Anaconda/ Miniconda:

`$ conda env create -f environment.yml`

`$ conda activate AccCyc_spyder`

`$ spyder`

The typical installation time should take ~a couple minutes (if Miniconda/ Anaconda is already installed).

## Dataset

The dataset folder can be downloaded from the following: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10902908.svg)](https://doi.org/10.5281/zenodo.10902908).

## Workflow

In the dataset folder, it contains the MPPT data for all the temperatures: 25, 35, 45, and 55C, and already filtered out for good, working cells. 

To run the data analysis, open `20240401_AcceleratedCycleAnalysis_Vmpp_04.py` on Spyder. It is split into a couple sections.
1. **Functions**: contains all the functions needed to run the analysis.
2. **Loading all parameters**: each batch of experiment (in particular temperature and cycle length), has 8 devices (which has 6 pixels or 'small cells'). Each device corresponds to a specific parameter, and it needs to be named. *Ulas* corresponds to the SAM-based cells, and *Zahra-Batch2* corresponds to the NiOx-based cells. We are loading all the MPP data for these cells in this section. The `limit_h_25C` corresponds to the maximum aging test length you would like to include in this analysis. `folder_run_name` corresponds to the name of folder name where you save all the data from the analysis.
3. **Load data, calculate energy yield/area per cycle**: this loads the data, calculate energy yield/ area under the MPP PCE curve, for each cycle. It also calculates the statistics for this data.
4. **Difference in 0 vs. 1000h**: this shows how the PCE and V MPP have changed between 0 to 1000h.
5. **Plotting part 1**: this section plots the area under various parameters for each batch, and overall.
6. **Regression of parameters and Arrhenius fitting**: to see the 'slope'/ 'degradation rate', we perform regression for all the data with various parameters, and look at the Arrhenius analysis as well.
7. **Comparable area across different T**: this fits the PCE MPP curve at each cycle, for all the batches (in 6 different functions). 
8. **Read all dataframes, decide withc one is the best**: .
9. **Find equivalent time/ backcalculation**: for all the PCE MPP curve already fitted, now, the equivalent time length across temperatures is calculated. 
10. **Boxplot of backcalculation**: this plots all the backcalculation/ equivalent time results together, across temperatures, for all the types of devices.

This whole run could take a couple of hours on "normal" computer, especially in step 6-9. Other than backcalculation, it takes < 1 hour to go through.

## Authors
| |  | 
|---|---|
|**Author(s)** | Noor Titan Putri Hartono |
|**Version** | 1.0/ December 2023  |   
|**E-mail(s)**   | titan dot hartono at helmholtz-berlin dot de  |
| | |

## Attribution
This work is under an BSD 2-Clause License License. Please, acknowledge use of this work with the appropriate citation to the repository and research article.

## Citation

    @software{noortitan_2023_8181602,
      author       = {noortitan},
      title        = {noortitan/AccCyc: First Release},
      month        = dec,
      year         = 2023,
      publisher    = {},
      version      = {v1.0},
      doi          = {},
      url          = {\url{}}
    }
