# dnn_contour_integration

This repository contains all the code and necessary files (datasets, model weights and psychophysics experiemnt) for the paper 'A feedforward mechanism for human-like contour integration'.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Description
This repository contains all the code and necessary files (datasets, model weights and psychophysics experiemnt) for the paper 'A feedforward mechanism for human-like contour integration'. The repository contains the following folders:

* contour_integ_stimuli
* contour_integ_models
* contour_integ_behavior


## Installation

Follow these steps to set up the repository and install all necessary dependencies:

```bash
# Clone this repository
git clone https://github.com/feziodoshi/dnn_contour_integration.git
cd dnn_contour_integration

# Install the dependencies from requirements.txt
pip install -r requirements.txt

# Install the additional GitHub dependency
pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git

# Create the relevant_files directory and subdirectories
mkdir -p relevant_files/contour_dataset
mkdir -p relevant_files/model_weights
mkdir -p relevant_files/psychophysics_experiment


# Extracting the files

# Contour Dataset - Training and Validation
wget http://example.com/model_training.tar -P relevant_files/contour_dataset
tar -xvf relevant_files/contour_dataset/model_training.tar -C relevant_files/contour_dataset
mv relevant_files/contour_dataset/model_training ../../

# Contour Dataset - Psychophysics
wget http://example.com/model_psychophysics.tar -P relevant_files/contour_dataset
tar -xvf relevant_files/contour_dataset/model_psychophysics.tar -C relevant_files/contour_dataset
mv relevant_files/contour_dataset/model_psychophysics ../../

# Model Weights
wget http://example.com/model_weights.tar -P relevant_files/model_weights
tar -xvf relevant_files/model_weights/model_weights.tar -C relevant_files/model_weights
mv relevant_files/model_weights/model_weights ../../

# Psychophysics Experiment
wget http://example.com/contour_exp1.tar -P relevant_files/psychophysics_experiment
tar -xvf relevant_files/psychophysics_experiment/contour_exp1.tar -C relevant_files/psychophysics_experiment
mv relevant_files/psychophysics_experiment/contour_exp1 ../../
```






## Usage
The repository contains the following files:

* field_stim_functions.py - python file containing all the relevant functions to generate the stimuli similiar to Field et al., 1993

* generate_sample.ipnb - notebook containing interactive code to visualize sample stimuli - contour stimuli with maks and information about all elements in the display

* generate_training_psychophyics.ipnb - notebook containing code to generate (a) training (b) validation and (c) psychophyics stimuli

* field_stim_functions.py - python file containing config information for (a) training (b) validation and (c) psychophyics stimuli


## License

Specify the license under which your project is distributed.