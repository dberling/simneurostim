# simneurostim
Simulate neural stimulation with biophysically detailed single neuron models (NEURON).

# Setting up the simulation environment
We suggest to embed this repository with simulation and model code inside your project's repository, e.g.:
you_project/
    simneurostim/
    your_scripts/
    your_workflows/

You can then follow the following steps to set up the simulation software:

# install miniconda

# create conda env from yml-file

* run "conda env create -f environment.yml"

# activate conda env

* run "conda activate simneurostim"

# compile mod files

* run "nrnivmodl simneurostim/model/mod/"

# add local python package via conda-develop

* run "conda-develop simneurostim/base-neurostim/"

# test installation:

* run simulation_playground.ipynb and see if it runs without errors.

# Errors due to separation of simulation & model code from this particular analysis

Errors which might appear and how to fix:
* old module name "optostim" may appear in code, replace by "neurostim"
* paths of neuron / cell models may be given for the old location ("model/hoc/...") but are now located at "simneurostim/model/hoc/..."

