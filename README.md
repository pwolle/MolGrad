# MolGrad - a Score-Based Model for Molecule Generation and Optimization

This repo contains the official implementation of the Jugend forscht project 2021 by [Paul Wollenhaupt](https://github.com/unitdeterminant).

## What is MolGrad?

MolGrad learns from drug databases in a completely unsupervised manner. It can use that knowledge to aid the drug development process, by (1) generating new, high quality molecules with high synthetic accessibility [1] and quantitative estimated druglikeliness [2] when searching for a lead compound and (2) further optimizing/editing specific traits of these compounds like their solubility.

MolGrad is the first method to score-based generative modelling for molecules. This approach has recently shown a lot of promise in other data domains [3-6] and the [paper](https://github.com/unitdeterminant/MolGrad/raw/main/paper.pdf) (German) argues, that with a few novel adjustments and a newly developed architecture, score-based modelling is a perfect fit for molecules.

A short explanation video is available [here](https://vimeo.com/553621029) (German).

### New Architecture

![transformer_gnn](https://user-images.githubusercontent.com/77510444/118408622-af6d9780-b686-11eb-9d9e-4d7426c7c281.png)

## Setup

Clone this repository if you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed, or download and unzip alternatively.  

```console
C:\...> git clone https://github.com/unitdeterminant/MolGrad.git
```

Install anaconda and add it to `PATH` like [this](https://docs.anaconda.com/anaconda/install/). Then navigate to the repository in your shell, create a new environment from the environment.yml file and activate it.

```console
C:\...\MolGrad-main> conda env create --name molgrad --file environment.yml
C:\...\MolGrad-main> conda activate molgrad
```

## Usage

To generate, edit and optimize molecules and reproduce the results from the paper start JupyterLab (included in the environment) and open `test.ipynb`. Two models, trained on the six-atom sized molecule subset of the gdb13 database [7].

```console
(molgrad) C:\...\MolGrad-main> jupyter lab
```

### Inspect Logs

Start tensorboard (also in the environment), open your browser and navigate to [http://localhost:6006/](http://localhost:6006/).

```console
(molgrad) C:\...\MolGrad-main> tensorboard --logdir logs --port 6006
```

### Retrain Models

Download the file called `gdb13.tgz` from official page [here](https://gdb.unibe.ch/downloads/), unzip it and move it to `data\raw\`. The folder structure should be:

```console
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

```console
(molgrad) C:\...\MolGrad-main> python trainscore.py --run_name "retrained"
(molgrad) C:\...\MolGrad-main> python trainprops.py --run_name "retrained"
```

## Refrences

To cite this work use:

```bibtex
@unpublished{
    Wollenhaupt2020MolGrad,
    author = "Paul Wollenhaupt",
    title  = "MolGrad: Moleküle generieren und optimieren mit KI",
    year   = 2020}
```

1. Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of cheminformatics, 1(1), 1-11.
2. Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012). Quantifying the chemical beauty of drugs. Nature chemistry, 4(2), 90-98.
3. Chen, N., Zhang, Y., Zen, H., Weiss, R. J., Norouzi, M., & Chan, W. (2020). WaveGrad: Estimating gradients for waveform generation. arXiv preprint arXiv:2009.00713.
4. Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. arXiv preprint arXiv:2105.05233.
5. Cai, R., Yang, G., Averbuch-Elor, H., Hao, Z., Belongie, S., Snavely, N., & Hariharan, B. (2020). Learning gradient fields for shape generation. arXiv preprint arXiv:2008.06520.
6. Niu, C., Song, Y., Song, J., Zhao, S., Grover, A., & Ermon, S. (2020, June). Permutation invariant graph generation via score-Based generative modeling. In International Conference on Artificial Intelligence and Statistics (pp. 4474-4484). PMLR.
7. Blum, L. C., & Reymond, J. L. (2009). 970 million druglike small molecules for virtual screening in the chemical universe database GDB-13. Journal of the American Chemical Society, 131(25), 8732-8733.
