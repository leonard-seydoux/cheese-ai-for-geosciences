# Artificial Intelligence and Machine Learning for Geosciences 

Material for the training course at the Barcelona Supercomputing Center, __5–7 November 2024__, delivered by:
- Léonard Seydoux (Assistant Professor, IPGP)
- Hugo Frezat (ChEESE postdoc, IPGP/CNRS)
- Alexandre Fournier (Senior professor, IPGP)
- Geneviève Moguilny (Research engineer, CNRS/IPGP).

<img src="logo.png" width=500>
 
ChEESE is the [Center of Excellence for Exascale in Solid Earth](https://cheese2.eu/). ChEESE has received funding from the European High Performance Computing Joint Undertaking (JU) and Spain, Italy, Iceland, Germany, Norway, France, Finland and Croatia under grant agreement No 101093038. The project aims to prepare European codes for the upcoming exascale supercomputers. The training is part of the ChEESE training program.
IPGP is the [Institut de Physique du Globe de Paris](https://www.ipgp.fr/), a French research institution dedicated to the study of Earth and planetary sciences.
BSC is the [Barcelona Supercomputing Center](https://www.bsc.es/), the Spanish national supercomputing center.

## Description

This training is aimed at geophysicists willing to develop an operational sense of AI, with a working knowledge of Python. Attendees with a lack of knowledge of Python will be provided with an upgrade notebook beforehand. Background in inverse problems, statistics or data assimilation is a plus. 

This repository contains the course material for the training, including the notebooks, the slides and the data. The material will be dropped in the repository as the course progresses (slides, notebooks, data, and solutions).

## Goals

The content of this course was partially taken from the master-level class _Earth Data Science_ of the institut de physique du globe de Paris taught by [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil), [Alexandre Fournier](https://www.ipgp.fr/~fournier/), [Éléonore Stutzmann](https://www.ipgp.fr/~stutz/) and [Antoine Lucas](http://dralucas.geophysx.org/). 

The goal of this course is to introduce students to the basics of scientific computing and to the use of Python for solving geophysical problems. The course mostly consists in practical sessions where students will learn how to use Python to solve problems related to the Earth sciences mith statistical and machine learning methods. The course and notebooks rely on the Python [scikit-learn](https://scikit-learn.org/stable/) library, [pandas](https://pandas.pydata.org/), [pytorch](https://pytorch.org/), and the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, Yoshua Bengio and Aaron Courville.

## Contents

The labs are self-explanatory and listed in order under the `labs` directory. Each lab is introduced in the series of slides under the `lectures` directory. An extended explanation explain each lab's goals, and the solutions to each exercice are also provided. 

## Installation

### Creating virtual environments 

In order to run the labs, several environments are required. We recommend to use the Anaconda package manager, althgouh other managers can work but are not tested by our team. Assuming you already have [installed Anaconda on your computer](https://docs.anaconda.com/anaconda/install/) and that you have access to a Unix shell, you can then install the two virutal environments `cheese-torch` and `cheese-jax` with the following commands:

```bash
conda create --yes --name cheese-torch --file requirements-torch.txt
conda create --yes --name cheese-jax --file requirements-jax.txt
```

Once executed, these two command lines must have created the two virtual environements. You can check it with running

```
conda env list
``` 

These environments will then be visible as availble Python kernels from your Jupyter notebooks. The `cheese-torch` environment is made for the 3 first notebooks, and the `cheese-jax` was created for the last one.

### Troubleshooting

Because these environments are GPU compatible, we anticipate several causes of trouble when installing the virtual environment. Generally speaking, Anaconda may have trouble managing the versions of the required libraries. In case the direct installation with the `requirements-*.txt` files fails, you can also try to run the following commands without specifying the required version (example for `cheese-torch`):

```
conda create --name cheese-torch
conda activate cheese-torch
conda install -c conda-forge -c pytorch matplotlib numpy pandas scikit-learn obspy laspy tqdm cartopy scipy seaborn pytorch torchvision jupyter ipykernel
```

The `cheese-jax` environment is more delicate to install since it strongly depends on the version of the NVIDIA driver. We here deliver the list of strictly required packages that you must have access to, and that you can try and install depending on your hardware:

- jax for NVIDIA GPU
- flax
- optax
- jupyter
- matplotlib
- numpy
- tqdm

Note that both Torch and JAX libraries also have a CPU-compatible version, which are easier to install but less computationally efficient.
