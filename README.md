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

# Download additional files and place them in the parent directory
wget http://example.com/somefile
mv somefile ../
wget http://example.com/anotherfile
mv anotherfile ../






## Usage
The repository contains the following files:

* field_stim_functions.py - python file containing all the relevant functions to generate the stimuli similiar to Field et al., 1993

* generate_sample.ipnb - notebook containing interactive code to visualize sample stimuli - contour stimuli with maks and information about all elements in the display

* generate_training_psychophyics.ipnb - notebook containing code to generate (a) training (b) validation and (c) psychophyics stimuli

* field_stim_functions.py - python file containing config information for (a) training (b) validation and (c) psychophyics stimuli


## License

Specify the license under which your project is distributed.