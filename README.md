<!--# GerNA-Cert >



## Overview

🚀 We introduce GerNA-Bind, a geometric deep learning framework that excels in predicting RNA-ligand binding specificity by integrating multi-modal RNA-ligand representations. GerNA-Bind achieves state-of-the-art performance, successfully identifying 19 compounds binding to oncogenic MALAT1 RNA through high-throughput screening. Wet-lab validation confirmed three compounds with submicromolar affinities, showcasing its potential for advancing RNA-targeted drug discovery.

![GerNA-Bind Overview](./image/GerNA-Bind.jpg)

## 🗂 Contents

- [Installation](#installation)
<!-- - [Data Preparation](#data-preparation)
- [Generate RNA Structure](#generate-rna-structure) -->
- [Data Preparation](#data-preparation)
  - [Dataset Description](#dataset-description)
  - [Generate RNA Structure](#generate-rna-structure)
  - [Data Processing](#data-processing)
  - [Dataset Download](#dataset-download)
- [Using GerNA-Bind](#using-gerna-bind)
  - [Model Training](#model-training)

## Installation
### If you prefer a faster setup, you can use the provided gernabind.yaml file:
```bash

conda env create -f gernabind.yaml -y
conda activate gernabind

```

### You can also install the environment either by following the step-by-step instructions below.
```bash
# Create a conda environment
conda create -y -n gernabind python=3.8
conda activate gerncert

#conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=12.2 -c pytorch -c nvidia
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install other dependencies 
pip install rna-fm==0.2.2
pip install ml_collections==0.1.1
pip install simtk==0.1.0
pip install openmm==8.1.1
pip install torchdrug==0.2.1
pip install torch_geometric==2.4.0
pip install equiformer-pytorch
pip install edl_pytorch==0.0.2
pip install rdkit==2023.9.5
pip install biopython==1.79
pip install pandas==1.5.3
pip install scikit-learn==1.2.2
pip install prody==2.4.1
```


## Data Preparation
We will update the usage of building datasets and how to build your own datasets for training soon.
### Dataset Description
Refer to the following guides for setting up datasets:
- RNA-Small Molecule Screening datasets: [Robin](https://pubmed.ncbi.nlm.nih.gov/36584293/) and [Biosensor](https://pmc.ncbi.nlm.nih.gov/articles/PMC9477273/)
- RNA-Ligand Complex Structure datasets: [Hariboss](https://hariboss.pasteur.cloud/)

### Generate RNA Structure
We use [RhoFold+](https://github.com/ml4bio/RhoFold) to generate RNA 3D Structure and [RNAfold](https://github.com/ViennaRNA/ViennaRNA/releases) (version: 2.5.1) to generate RNA 2D structure.

### Data Processing
You can process data through the following steps:
```bash
python data_utils/process_data.py 
```
And the processed data will be saved in ./data folder as "new_data.pkl" file.

### Dataset Download
We process the processed data with Robin & Biosensor dataset. You can download the processed data from [Zenodo](https://zenodo.org/records/17770641).

We also provide the scripts for downloading the dataset.
Simply run:
```bash
bash download_dataset.sh
```
preprocess data：
```bash
bash convert_to_fast_data.py
```


## Using GerNA-Bind
### Model Training
We provide the training scripts that you can train the model yourself.
```bash
python train_model.py 
```

## Acknowledgements

Our work builds upon [EquiFormer](https://github.com/atomicarchitects/equiformer), [Evidential Deep Learning](https://github.com/aamini/evidential-deep-learning), [MONN](https://github.com/lishuya17/MONN), [RNA-FM](https://github.com/ml4bio/RNA-FM), [RhoFold](https://github.com/ml4bio/RhoFold), and [TankBind](https://github.com/luwei0917/TankBind). Thanks for their excellent work and open-source contributions.
