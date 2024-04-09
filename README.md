# bsg_seismology
This is a collection of codes / routines that will be used to detect and characterise asteroseismic signatures in pulsating blue super giant stars observed with TESS.  

### Code author:
  - Cole Johnston

### Project 1

  - Lead:
    - Linhao Ma

  - Supervisors:
    - Cole Johnston
    - Earl Bellinger
    - Selma de Mink
    - Jim Fuller
  - Published as: 
    - https://ui.adsabs.harvard.edu/abs/2023arXiv231019546M/abstract

### Project 2: 
  - Lead: 
    - Cole Johnston
  - Collaborators:
    - Selma de Mink
    - Earl Bellinger

## Notebooks

I'll keep a few example notebooks for how to use the various functions in the repo.

  - example_01: At the moment, the first example uses functions stripped from LATTE (by N. Eisner) that are modified to work for our specific use case. 
  - example_02: Notebook that fits the power excess for a given BSG using a simple Harvey profile + white noise + gaussian excess component


## Installation

I've created an environment held in bsg.yml. This has all of the packages required for this project and they all have been checked for 
the appropriate dependencies. To install this environment, you must first EDIT THE PREFIX at the bottom of the yml file to point to your
installation of (ana/mini)conda. After you have modified this, you can you can issue:

conda env create -f bsg.yml

to create and populate the virtual environment. After this, you can launch the environment using either:

source activate bsg

 ### OR

conda activate bsg

depending on how you've set up (ana/mini)conda.

### Non-conda installs
Furthermore, we need to use the latest development version of astroplan. This can be accomplished by downloading it directly from 
github by issuing:

git clone https://github.com/astropy/astroplan

python setup.py build
python setup.py install

This will install the package in the relevant location so long as you do this while the bsg environment is activated.

Finally, we will need the pythia package:
  git clone https://github.com/colej/pythia

The package has its own install instructions that can be found on the page.
