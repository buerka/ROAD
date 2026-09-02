# `ROAD` 🛣️ : The Radio Observatory Anomaly Detector

A repository containing the implementation of the paper entitled [The ROAD to discovery: machine learning-driven anomaly detection in radio astronomy spectrograms](https://arxiv.org/abs/2307.01054)


## Installation 
Install conda environment by:
``` 
    conda create --name road python=3.9.7
``` 
Run conda environment by:
``` 
    conda activate road
```

Install the appropriate pytorch version:
``` 
    conda install pytorch torchvision torchaudio pytorch-cuda=<VERSION> -c pytorch -c nvidia
``` 

Install dependancies by running:
``` 
    pip install -r requirements
```

## Dataset  
You will need to download the [ROAD dataset](https://zenodo.org/record/8028045) and specify the its path using `-data_path` command line option.

### Low-memory dataset access

The loader keeps the paper's data protocol unchanged while avoiding repeated
in-memory copies.  It catalogs the published HDF5 file by sample ID, reports
the 7,053 physical rows and 6,708 unique observations, and then exposes the
same 6,301-sample experiment view as the original `_join(...,
compound=False)` implementation.  In particular, the 407 compound-label
observations remain outside the model-facing view because adding them to the
single-label loss would change the experiment.

On the first run, ROAD normalizes that experiment view in bounded HDF5 blocks
and creates one shared, read-only-on-disk NumPy cache.  The cache deliberately
uses the original float64 normalization and NHWC backing layout, so values,
tensor strides, sample ordering, split behavior, and contamination sampling
remain compatible with the published implementation.  The five Dataset
objects retain only index arrays and reuse this cache.

Allow approximately 12.31 GiB of free disk for the cache.  The first run takes
longer; later runs reuse it.  By default it is written to
`<model_path>/.road_cache`, or beside the HDF5 file when `-model_path` is not
set.  A different directory can be selected explicitly:

```
python main.py -data_path /path/to/ROAD_dataset.h5 \
    -model_path /path/to/run-root \
    -data_cache_path /path/to/road-cache
```

Cache identity includes the HDF5 path, size, modification time, normalization
amount, record order, and cache format version.  Concurrent first runs wait on
one cache builder instead of generating duplicate 12 GiB temporary files.



## Replication of results in paper 
Run the following to replicate the results for the resnet34 used in the paper
```
    ./experiments/final_model.sh
```
or to run for all backbones 
```
    ./experiments/test.sh
```
Alternatively the [model weights](https://zenodo.org/record/8060501) can be downloaded and specified using the  `-model_name` and `-model_path` flags.


## Labelling with label-studio:
The labelling interface is based on [label-studio](https://labelstud.io/). To get the label server running for the LOFAR_AD project, run the following:
```
  label-studio start LOFAR_AD --sampling uniform &
```
and
```
./webserver /home/mmesarcik/data/LOFAR/compressed/LOFAR_AD/LOFAR_AD_v1/ *.png files 8081
````

## Licensing
Source code of ROAD is licensed under the MIT License.
