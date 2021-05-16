# MolGrad - a Score-Based Model for Molecule Generation and Optimization
This repo contains the official implementation of the Jugend forscht project 2021 by [Paul Wollenhaupt](https://github.com/unitdeterminant).

## What is MolGrad?
MolGrad learns from drug databases in a completely unsupervised manner. It can use that knowledge to aid the drug development process, by (1) generating new, high quality molecules with high synthetic accessibility and quantitative estimated druglikeliness when searching for a lead compound and (2) further optimizing/editing specific traits of these compounds like their solubility.

MolGrad is the first method to score-based generative modelling for molecules. This approach has recently shown a lot of promise in other data domains and the [paper](https://github.com/unitdeterminant/MolGrad/raw/main/paper.pdf) (German) argues, that with a few novel adjustments and a newly developed architecture, score-based is a perfect fit for molecules.

A short explanation video is available [here](https://vimeo.com/546206651) (German).

### New Architecture
![transformer_gnn](https://user-images.githubusercontent.com/77510444/118408189-b7c4d300-b684-11eb-9e48-b43f807badb3.png)


## Setup
Clone this repository if you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed, or download and unzip alternatively.  
```
C:\...> git clone https://github.com/unitdeterminant/MolGrad.git
```

###
Install anaconda like [this](https://docs.anaconda.com/anaconda/install/). Then navigate to the repository in your shell, create a new environment from the environment.yml file and activate the environment.

```
C:\...\MolGrad-main> conda env create --name molgrad --file= environment.yml
C:\...\MolGrad-main> conda activate molgrad
```

## Usage
To generate, edit and optimize molecules and reproduce the results from the paper start JupyterLab (included in the environment) and open `test.ipynb`. Two models, trained on the six-atom sized molecule subset of the gdb13 database.

```
(molgrad) C:\...\MolGrad-main> jupyter lab
```

### Inspect Logs
Start tensorboard (also in the environment), open your browser and navigate to http://localhost:6006/.
```
(molgrad) C:\...\MolGrad-main> tensorboard --logdir logs --port 6006
```

### Retrain Models
Download the file called `gdb13.tgz` from official page [here](http://gdbtools.unibe.ch:8080/cdn/gdb13.tgz), unzip it and move it to `data\raw\`. The folder structure should be:

```
MolGrad-main
├─── data
│     ├─── raw
│     │     ├─── 1.smi
│     │     ├─── 2.smi
│     │     ⋮

│     │     ├─── 6.smi
│     │     ⋮
│     │     └─── 13.smi
│     │
│     └─── __init__.py
⋮
```



To retrain the score model run `trainscore.py` and `trainprops.py` for the property regression model. New logs will show up in the tensorboard tab. To use these models in the notebook change `score_run_name` and `score_run_name` to `"retrained"`.

```
(molgrad) C:\...\MolGrad-main> python trainscore.py --run_name "retrained"
(molgrad) C:\...\MolGrad-main> python trainprops.py --run_name "retrained"
```
