# pyCGE
This repository provide an educational package to develop computational general equilibrium and perform counterfactual analysis.

PyCGE is a Python 3 implementation of routines for calibrating various computational general equilibrium and perfoming counterfactual analysis. The package lists various models in the literature and show how people can calibrate and perform counterfactual experments. This package was created by Yuta Watabe. 

# Installation
The PyCGE package is developed in python 3.9.6. The exact requirement of packages are listed in poetry.lock.

We recommend to use VScode + docker to use this package. In this way, you do not need a python in your computer. Python and other related packages are installed in the docker, which is separated from other namespaces. To use VScode + docker, you need to follow these three steps:
  - Install docker desktop 
  - Install VScode
  - Activate VScode and install extension "Remote - Containers"

After the installation:
  - Clone this repository
  - Open this repository
  - It should automatically start building a container. It will take a while
  - After the container is build, you can start running a code
