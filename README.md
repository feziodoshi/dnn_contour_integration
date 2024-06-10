# dnn_contour_integration

This repository contains all the code and necessary files (datasets, model weights and psychophysics experiemnt) for the paper 'A feedforward mechanism for human-like contour integration'.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Description
This repository contains all the code and necessary files (datasets, model weights and psychophysics experiemnt) for the paper 'A feedforward mechanism for human-like contour integration'. The repository contains the following folders:

* contour_integ_stimuli  - Use it to generate all contour stimuli
* contour_integ_models   - Use it to train different kiuds of feedforward contour readout models and analyse the internal activations. Additional code is provided to compute saliency maps and sensitivity to contour alignment
* contour_integ_behavior - Use it to view unprocessed humam behavioral data. Code is provided to perform psychophysics on any trained model and compare it with humans to measure human-model alignment
* manuscript_figures     - Use it to generate all figures from the manuscript. To replicate the exact same figures, you will need access to all the model weights and the contour dataset (provided in the relevant files folder)
* relevant_files         - Contains all the necessary files - contour dataset (training, validation, and psychopbhysics), model weights (pretrained dnn backbone weights and contour readout weights), and code for psychophysics experiemnet



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


# Extracting the files
mkdir -p relevant_files

# 1. Contour Dataset - Training, Validation, and Psychophysics dataset
wget -O relevant_files/temp_download.zip "https://www.dropbox.com/scl/fo/rxfzsqhkv6mw8gif7d15w/AIpDy7XSYfnlO1i8HTfBiiA?rlkey=21ifwapf46mflb25iaiy6ne2f&st=77k0pm0z&dl=1"
mkdir -p relevant_files/contour_dataset
unzip relevant_files/temp_download.zip -d relevant_files/contour_dataset/

# 2. Model Weights
wget -O relevant_files/temp_download.zip "https://www.dropbox.com/scl/fo/ambt5caokz4gybg3n19yt/AAwxCcW4ic9dw8qPL6YKYsE?rlkey=expo3ewzxohhcpj6s3t13110q&st=cd2v5lzd&dl=1"
mkdir -p relevant_files/contour_dataset
unzip relevant_files/temp_download.zip -d relevant_files/contour_dataset/


# 3. Psychophysics Experiment
wget -O relevant_files/temp_download.zip "https://www.dropbox.com/scl/fo/6x6vovfkkbmjujock9px0/AJCvNGJje1RgPQUvCOoFPq0?rlkey=48lobiml61e2m1v87rr2kayh7&st=6nuet6r9&dl=1"
mkdir -p relevant_files/psychophysics_experiment
unzip relevant_files/temp_download.zip -d relevant_files/psychophysics_experiment/
rm -rf relevant_files/temp_download.zip

# If you plan on using the existing datasets and the model weights to reconstruct manuscript figures:
# Step 1: Extract the file from the relevant files folder
tar -xvf relevant_files/<weights or dataset folder>/<filename> -C relevant_files/<weights or dataset folder>
# Step 2: Navigate to the folder of interest and move it to the parent directory (using the mv command):
# a) For model weights find the model weights folder and move it to the parent directory 
# b) For datasets you can move the model-training or model-psychophysics dataset in the relevant location. However you will have to add the absolute paths in all config files provided in each subdirectory to run the notebooks
```





## Usage
The repository contains the following files:

* field_stim_functions.py - python file containing all the relevant functions to generate the stimuli similiar to Field et al., 1993

* generate_sample.ipnb - notebook containing interactive code to visualize sample stimuli - contour stimuli with maks and information about all elements in the display

* generate_training_psychophyics.ipnb - notebook containing code to generate (a) training (b) validation and (c) psychophyics stimuli

* field_stim_functions.py - python file containing config information for (a) training (b) validation and (c) psychophyics stimuli


## License

Specify the license under which your project is distributed.