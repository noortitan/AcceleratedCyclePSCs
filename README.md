# Accelerated cycle aging test analysis for perovskite solar cells (AcceleratedCyclePSCs)

Version: 2023-11-29

This documentation is prepared as the workflow to accompany the following study:

**"Accelerating Light-Cycled Aging for Testing Perovskite Solar Cells Stability"**

*(INSERT JOURNAL INFO)*

Noor Titan Putri Hartono (1), Artem Musiienko (1), Ulas Erdil (1), Zahra Loghman Nia (1), Mark Khenkin (1), Hans Köbler (1), Johannes Beckedahl (1), Rutger Schlatmann (1), Carolin Ulbrich (1), Antonio Abate (1)

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

## Workflow

In the dataset folder, it contains the MPPT data for all the temperatures: 25, 35, 45, and 55C, and already filtered out for good, working cells. 

To run the data analysis, open `20231201_AcceleratedCycleAnalysis_Vmpp_03.py` on Spyder. It is split into a couple sections.
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















## What does the script do?

This script helps to "unpack" the `Data.csv` file generated from the aging test run. The `.csv` file is quite complicated because it contains all the measurements (MPPT, JV, etc.), with various trigger codes to identify which specific measurement it is. If you are curious about this, you can also read Hans Köbler's PhD Thesis :)  

Besides "unpacking" the `Data.csv` file, the script(s) also do the following things:
1. Extract the data, save them as separate `.csv` files for each pixel.
2. Connect the parameters to specific devices (i.e. if you have some 'repeated' samples/devices in the same aging batch).
3. Plot the overview of the data.

For your analysis, you may also:
1. Filter out some pixels which behave very differently from the rest of the pixels (might be due to loose connection, bad pixel, etc.). The average MPP data will also be calculated from this ()
2. Identify best pixels for each parameter. The script also plots this.
3. Calculate performance ratio (ratio of the energy yield for every 24 hours, across parameters, normalized to 48 hours performance --# hours can also be changed, if you wish).

## What does the folder contain?

The script folder contains the following:
1. `load_mpp_hysprint.py`: load the MPPT data, extract them into separate `.csv` files, for each pixel.
2. `req.txt`: the list of packages to be installed to run the script.
3. `AgingTest_ExperimentalMethod_Paper_Template.docx`: the experimental method template for your paper.

The data folder generated from the aging test contains the following:
1. `Data.csv`: the MPPT data (loaded with the `load_mpp_hysprint.py` script file).
2. `IVData` folder: the JV sweep data.
3. `Info.txt`: the information of the aging test setting, you can find the technical details regarding the shutter mode, experiment type, IV V min, IV V max, IV V step, IV interval, MPPT interval.
4. `Aging Request <your name>.xlsx`: the Excel file you submit, that list out your parameters, your device stack, etc.

The `extracted` folder generated from the data processing contains the following:
1. `JVfor`, `JVrev`, `JVfor_filtered`, `JVrev_filtered`: JV parameters from JV forward, reverse, filtered data.
2. `MPPT`, `MPPT_all_pixels`, `MPPT_filtered`, `MPPT_filtered_best`, `MPPT_filtered_mean`, `MPPT_performance_ratio`: MPPT data unfiltered, figures for all pixels, MPPT filtered, best pixel, and average/ mean. There's also the folder for calculated MPPT performance ratio.
3. `best_pixel.txt`, `cell_parameters_list.txt`, `filtered_out_list.txt`: information files, to store all the best pixels, cell parameters, and filters listed during the data processing.

## Installation and requirements

The script is on Python (there's also the MATLAB version, written by Hans). I usually use Spyder (run through Anaconda) to run it, but please feel free to use anything to run the script.

To use Anaconda to run the script, follow the following steps:
1. Install Anaconda (or the 'compact version', MiniConda), you may download it from here: https://docs.conda.io/en/latest/miniconda.html.
2. Open the Anaconda/ MiniConda prompt. 
3. Go to the folder containing the script files.
`$ cd AgingDataProcessing`
3. Create a virtual environment with `<environment_name>` name and install the necessary packages (using the `req.txt` file).
`$ conda create -n <environment_name> --file req.txt`
4. Activate the virtual environment.
`$ conda activate spyder-env`
5. Open Spyder.
`$ spyder`

## Running the scripts

**IMPORTANT NOTE: Python indexing starts with 0, not with 1. That means, parameters, pixel numbers, etc. start with 0. Pixel numbers go from 0-5, instead of 1-6. Sample numbers go from 0-7, instead of 1-8.**

### `load_mpp_hysprint.py`

1. Scroll down the file, skip the whole functions section, go to the "Running: Initialization: Folder, Cell area, & Load file" section. Change the folder location of your data (`folder_loc`), the folder selected name containing your dataset (`folder_selected`), your cell area (`cell_area`), your type of measurement (`type_measurement`) which only takes 2 types of input: `'cycle'` or `'continuous'`, and if it's a cycle measurmenet, you should also input the `measurement_cycle` which correspnds to the length of the cycle (if it's 12-12h cycle, the `measurement cycle = 12`, and if it's 6-6h cycle, the `measurement cycle = 6`). After filling in all the information, run this section (after running the function section, of course).
2. In the next section "Running: Parameters, Pixel filters", there are 3 variables need to be defined: `cell_param`, `pixel_filter`, `best_pixel`. `cell_param` is the dictionary containing your parameters. For example, you have 2 parameters in your batch: C60 and C60_SAM. You can enter the `cell_param` variable as the following:
`cell_param = {0 : [0,'C60'],
              1 : [1,'C60_SAM'],
              2 : [0,'C60'],
              3 : [1,'C60_SAM'],
              4 : [0,'C60'],
              5 : [1,'C60_SAM'],
              6 : [0,'C60'],
              7 : [1,'C60_SAM']}`
This means, that you have 8 samples (0-7), where samples 0, 2, 4, 6 have parameter 0, 'SAM', and samples 1, 3, 5, 7 have parameter 1, 'C60_SAM'. Make sure that you are consistent in naming the number of parameters (0 & 1 in this case) with the name of parameters ('C60' and 'C60_SAM' in this case). 
3. Run the next two sections "Running: Save CSV for all samples and pixels", and "Running: group parameters + filter + resample + interpolate +  average + find best pixel + performance ratio". A couple plots will be generated.
4. Look at the plots, and decide on `pixel_filter` and `best_pixel`.
5. For the `pixel_filter`, the example is as following:
`pixel_filter = {0: [2],
                1: [],
                2: [5],
                3: [],
                4: [5],
                5: [],
                6: [1,2,3,5],
                7: []}`
The left side/ key of the dictionary corresponds to the device/ sample number, and the right side/ value of the dictionary corresponds to the list of pixels we are filtering out. In example above, it translates to, we are filtering out: pixel 2 in device 0, pixel 5 in device 2, pixel 5 in device 4, and pixels 1,2,3,5 in device 6. Adjust these numbers according to your observation from seeing the plots.
6. For the `best_pixel`, the example is as following:
`best_pixel = {0: [],
              1: [],
              2: [5],
              3: [4],
              4: [],
              5: [],
              6: [],
              7: []}`
The left side/ key of the dictionary corresponds to the device/ sample number, and the right side/ value of the dictionary corresponds to the pixel number which shows the best performance. In example above, it translates to, the best pixel for each parameters is in pixel 5 device 2, and pixel 4 device 3, respectively.
7. The process of choosing which pixels to filter out, which pixel is the best pixel for a specific parameter is an iterative process, and will take a couple tries before the results are satisfactory.
8. All the extracted files area available in `extracted` folder.

### `load_JV_ageing.py`

This particular file was designed to run **cycled temperature, cycled illumination** experiment with large area, to check (and generate plot for) the energy yield result, and the temperature during the cycling process. Example experiment: `20231023_103103_Vaso-3`.

The current version of this is still raw/ not intended for a general analysis protocol. You may take a look and see it yourself, if you're curious. 

## Attribution

This work is under an BSD 2-Clause License License. Please, acknowledge use of this work with the appropriate citation to the repository and research article.

## Citation

    @software{noortitan_agingtest,
      author       = {Noor Titan Putri Hartono and  Michael Götte and Hans Köbler},
      title        = {Perovskite Solar Cells Aging Test Data Processing Software},
      month        = nov,
      year         = 2023,
    }