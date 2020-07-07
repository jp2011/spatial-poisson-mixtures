spatial-poisson-mixtures
==============================

This repository accompanies the following paper:


Povala, Jan, Seppo Virtanen, and Mark Girolami. 2020. ‘Burglary in London: Insights from Statistical Heterogeneous 
Spatial Point Processes’. Journal of the Royal Statistical Society. Series C (Applied Statistics) forthcoming. 
https://arxiv.org/abs/1910.05212.


These codes have been tested to work on Ubuntu and OSX. It requires Python 3.

## Steps to execute the program

Note that, all the commands below need to be run from the root of this repository.

### 1. Conda environment
Create a conda environment with the required libraries and activate it.
```bash
conda env create -f conda-python-env.yml
conda activate london-crime-mixtures
```

### 2. Compile the C code for resampling mixture allocations
```bash
python src/z_sampler/setup.py  build_ext --build-lib ./
```

### 3. Running the models

Codes for the three models that we consider in the paper are available in the `./src/models` folder. The naming 
conventions follow the variable names in the paper.

Running any of the three scripts as described below will produce regular snapshots of the respective model. The snapshots
are saved in a `.npy` file with the name based on the model context and the UID. These snapshots can the be processed
using the `src/experiment/results_processing.py` script. The attached notebook called `paper-results.ipynb` utilises
these scripts to produce figures in the paper.

If the scripts are run with `--verbose` flag, they will regularly inform the user about the state of the inference 
algorithm. One of the most useful information it gives is the acceptance ratio for each HMC sample as well as the 
'momentum' and the 'potential' parts of the ratio. Also, it regularly plots the traceplots for the quantities it samples.
This gives an indication if something is very wrong. More chains are recommended to be run if higher degree of belief 
of convergence is sought. 

#### Common parameters

1. `--model`: specifies the YAML filename with the configuration of the covariates. The YAML file needs to be placed
inside `./models/config/` directory.
2. `--type`: specifies the crime type. The data provides has counts for other crime types as well, but we only work 
with burglary.
3. `--resolution`: the provided data file has only 400m resolution so this is the only valid option. If it is to be
changed then it needs to regenerated as described below.
4. `--uid`: identifier used to generate file names for the inference snapshots. This uid(s) need to be supplied when
processing the results using `src/experiment/results_processing.py`.


#### LGCP model
```bash
python -m src.models.lgcp_matern --verbose --year 12013-122015 --type burglary --resolution 400 --model_name burglary_raw_1  --uid "LGCP_MODEL_UID"
```

#### Block mixtures with block dependence
```bash
python -m src.models.block_mixture_gp_softmax --verbose --type burglary --model_name burglary_raw_4  --resolution 400  --block_type msoa --year 12013-122015  --num_mixtures 3 --lengthscale 1000  --uid "BLOCK_MIX_GP_SOFTMAX_UID"
```

#### Block mixtures with independent blocks
```bash
python -m src.models.block_mixture_flat --verbose --year 12013-122015 --type burglary --resolution 400 --model_name burglary_raw_1   --num_mixtures 3 --block_type msoa --uid "BLOCK_MIXTURE_MODEL_UID"
```

## Producing plots, summaries, 
The notebook in `./notebooks/paper-results.ipynb` has codes to produce summaries and plots after the models have been
run. To a large extent, the notebook utilises `src/experiment/results_processing.py` script to produce the plots or
the supporting data for the plots. Some of the plots are `eps`, but many of them are standalone LaTeX files which after
compiling with LaTeX can be converted to a desired format. PAI/PEI section produces `.dat` tabular data file which can
be rendered by gnuplot, Excel, etc.

The notebook also has additional plotting that were used for diagnostics, e.g. traceplots for the samples.

## Platform-specific issues
To be able to correctly link the code to the required libraries on a MAC, you should run this command:
```bash
export CPATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/
```

## Creating custom datasets

One might want to create datasets that cover different time period, or use different spatial resolution. To do that,
one can utilise the script `src/data/r-spatial-data-extensive-retriever.R`. For example, one may want to run:
```R
./src/data/r-spatial-data-extensive-retriever.R --resolution 300 --startdate "2013-01-01" --enddate "2015-12-31"
```

Note that this might take time as for many covariates it downloads them from the internet. Not all data sources will 
work out of the box. For example, point of interest data, which I have obtained form Ordnance Survey data must be 
downloaded by the user separately and the path to such file must be specified in `r-spatial-data-extensive-retriever.R`.
Either, do not use POI data, or provide an alternative source and adjust the POI retrieval functions accordingly.

One might also want to process new crime data as reported on `police.uk` portal. To do so, place the downloaded CSVs
into `./data/raw/crime/` folder and run `./src/data/create_crime_db.py` script. This will create SQLite database that
the R script above used.
