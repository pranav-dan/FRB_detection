# FETCH


[![DOI](https://zenodo.org/badge/165734093.svg?style=flat-square)](https://zenodo.org/badge/latestdoi/165734093)
[![issues](https://img.shields.io/github/issues/devanshkv/fetch)](https://github.com/devanshkv/fetch/issues)
[![forks](https://img.shields.io/github/forks/devanshkv/fetch)](https://github.com/devanshkv/fetch/network/members)
[![stars](https://img.shields.io/github/stars/devanshkv/fetch)](https://github.com/devanshkv/fetch/stargazers)
[![GitHub license](https://img.shields.io/github/license/devanshkv/fetch)](https://github.com/devanshkv/fetch/blob/master/LICENSE)
[![HitCount](http://hits.dwyl.com/devanshkv/fetch.svg)](http://hits.dwyl.com/devanshkv/fetch)
[![arXiv](https://img.shields.io/badge/arXiv-1902.06343-brightgreen.svg)](https://arxiv.org/abs/1902.06343)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


fetch is Fast Extragalactic Transient Candidate Hunter. It has been detailed in the paper [Towards deeper neural networks for Fast Radio Burst detection](https://arxiv.org/abs/1902.06343).

This is the `tensorflow>=2` version of the fetch, if you are looking for the older tensorflow version click [here](https://github.com/devanshkv/fetch/archive/0.1.8.tar.gz).

Install 
---
    git clone https://github.com/devanshkv/fetch.git
    cd fetch
    pip install -r requirements.txt
    python setup.py install

The installation will put `predict.py` and `train.py` in your `PYTHONPATH`.

Usage
---
To use fetch, you would first have to create candidates. Use [`your`](https://thepetabyteproject.github.io/your/) for this purpose, [this notebook](https://thepetabyteproject.github.io/your/ipynb/Candidate/) explains the whole process. Your also comes with a command line script [`your_candmaker.py`](https://thepetabyteproject.github.io/your/bin/your_candmaker/) which allows you to use CPU or single/multiple GPUs. 

To predict a candidate h5 files living in the directory `/data/candidates/` use `predict.py` for model `a` as follows:

    predict.py --data_dir /data/candidates/ --model a
        
To fine-tune the model `a`, with a bunch of candidates, put them in a pandas readable csv, `candidate.csv` with headers 'h5' and 'label'. Use

    train.py --data_csv candidates.csv --model a --output_path ./
        
This would train the model `a` and save the training log, and model weights in the output path.

Example
---

Test filterbank data can be downloaded from [here](http://astro.phys.wvu.edu/files/askap_frb_180417.tgz). The folder contains three filterbanks: 28.fil  29.fil  34.fil.
Heimdall results for each of the files are as follows:

for 28.fil

    16.8128	1602	2.02888	1	127	475.284	22	1601	1604
for 29.fil

    18.6647	1602	2.02888	1	127	475.284	16	1601	1604
for 34.fil

    13.9271	1602	2.02888	1	127	475.284	12	1602	1604 

The `cand.csv` would look like the following:

    file,snr,stime,width,dm,label,chan_mask_path,num_files
    28.fil,16.8128,2.02888,1,475.284,1,,1
    29.fil,18.6647,2.02888,1,475.284,1,,1
    34.fil,13.9271,2.02888,1,475.284,1,,1

Running `your_candmaker.py` will create three files:

    cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_13.92710.h5
    cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_16.81280.h5
    cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_18.66470.h5

Running `predict.py` with model `a` will give `results_a.csv`:

    ,candidate,probability,label
    0,cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_18.66470.h5,1.0,1.0
    1,cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_16.81280.h5,1.0,1.0
    2,cand_tstart_58682.620316710374_tcand_2.0288800_dm_475.28400_snr_13.92710.h5,1.0,1.0
    
Training Data
---

The training data is available at [astro.phys.wvu.edu/fetch](http://astro.phys.wvu.edu/fetch/).

## Citating this work
___

If you use this work please cite:

    @article{Agarwal2020,
      doi = {10.1093/mnras/staa1856},
      url = {https://doi.org/10.1093/mnras/staa1856},
      year = {2020},
      month = jun,
      publisher = {Oxford University Press ({OUP})},
      author = {Devansh Agarwal and Kshitij Aggarwal and Sarah Burke-Spolaor and Duncan R Lorimer and Nathaniel Garver-Daniels},
      title = {{FETCH}: A deep-learning based classifier for fast transient classification},
      journal = {Monthly Notices of the Royal Astronomical Society}
    }
    @software{agarwal_aggarwal_2020,
      author       = {Devansh Agarwal and
                      Kshitij Aggarwal},
      title        = {{devanshkv/fetch: Software release with the 
                       manuscript}},
      month        = jun,
      year         = 2020,
      publisher    = {Zenodo},
      version      = {0.1.8},
      doi          = {10.5281/zenodo.3905437},
      url          = {https://doi.org/10.5281/zenodo.3905437}
    }
