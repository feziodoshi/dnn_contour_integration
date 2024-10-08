# A feedforward mechanism for human-like contour integration
***Fenil R. Doshi, Talia Konkle, and George A. Alvarez***  
*Department of Psychology, Harvard University, Cambridge, Massachusetts*  
*Kempner Institute for the Study of Natural and Artificial Intelligence, Harvard University, Cambridge, Massachusetts*



This repository contains all the code and necessary files (datasets, model weights and psychophysics experiemnt) for the paper 'A feedforward mechanism for human-like contour integration'.

# Abstract
Deep neural network models provide a powerful experimental platform for exploring core mechanisms underlying human visual perception, such as perceptual grouping and contour integration — the process of linking local edge elements to arrive at a unified perceptual representation of a complete contour. Here, we demonstrate that feedforward, nonlinear convolutional neural networks (CNNs), such as Alexnet, can emulate this aspect of human vision without relying on mechanisms proposed in prior work, such as lateral connections, recurrence, or top-down feedback. We identify two key inductive biases that give rise to human-like contour integration in purely feedforward CNNs: a gradual progression of receptive field sizes with increasing layer depth, and a bias towards relatively straight (gradually curved) contours. While lateral connections, recurrence, and feedback are ubiquitous and important visual processing mechanisms, these results provide a computational existence proof that a feedforward hierarchy is sufficient to implement gestalt "good continuation" mechanisms that detect extended contours in a manner that is consistent with human perception.

![](manuscript_figures/f6.png)



## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
- [Contributors](#contributors)
- [Citation](#citation)
- [License](#license)



## Description
This repository contains all the code and necessary files (datasets, model weights, and psychophysics experiments) for the paper 'A feedforward mechanism for human-like contour integration'. The repository contains the following folders:

* contour_integ_stimuli  - Use it to generate all contour stimuli.
* contour_integ_models   - Use it to train different kinds of feedforward contour readout models and analyze the internal activations. Additional code is provided to compute saliency maps and sensitivity to contour alignment.
* contour_integ_behavior - Use it to view unprocessed human behavioral data. Code is provided to perform psychophysics on any trained model and compare it with humans to measure human-model alignment.
* manuscript_figures     - Use it to generate all figures from the manuscript. To replicate the exact same figures, you will need access to all the model weights and the contour dataset (provided in the relevant files folder).
* relevant_files         - Contains all the necessary files - contour dataset (training, validation, and psychophysics), model weights (pretrained DNN backbone weights and contour readout weights), and code for the psychophysics experiment.



## Installation

Follow these steps to set up the repository and install all necessary dependencies:

```bash
# Clone this repository
git clone https://github.com/feziodoshi/dnn_contour_integration.git
cd dnn_contour_integration

# Create a virtual environment – https://realpython.com/python-virtual-environments-a-primer/ – for the project
python3 -m venv env
source env/bin/activate

# Install the dependencies from requirements.txt
pip install -r requirements.txt
# Install the additional GitHub dependency
pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git
```


If you plan on using the same contour datasets or the pre-trained contour-readout models, follow these steps to extract the relevant files:

```bash
# Extracting the files
mkdir -p relevant_files

# 1. Contour Dataset - Training, Validation, and Psychophysics dataset
wget -O relevant_files/temp_download.zip "https://www.dropbox.com/scl/fo/rxfzsqhkv6mw8gif7d15w/AIpDy7XSYfnlO1i8HTfBiiA?rlkey=21ifwapf46mflb25iaiy6ne2f&st=77k0pm0z&dl=1"
mkdir -p relevant_files/contour_dataset
unzip relevant_files/temp_download.zip -d relevant_files/contour_dataset/

# 2. Model Weights
wget -O relevant_files/temp_download.zip "https://www.dropbox.com/scl/fo/ambt5caokz4gybg3n19yt/AAwxCcW4ic9dw8qPL6YKYsE?rlkey=expo3ewzxohhcpj6s3t13110q&st=cd2v5lzd&dl=1"
mkdir -p relevant_files/model_weights
unzip relevant_files/temp_download.zip -d relevant_files/model_weights/

# 3. Additional Model Weights
wget -O relevant_files/temp_download.zip "https://www.dropbox.com/scl/fi/rjpp4r3hnj6usk63v3vi4/additional_model_weights.tar?rlkey=viflers5nulq0f95sd0wtgeq6&st=ualxbxxg&dl=1"
mkdir -p relevant_files/additional_model_weights
unzip relevant_files/temp_download.zip -d relevant_files/additional_model_weights/


# 4. Psychophysics Experiment
wget -O relevant_files/temp_download.zip "https://www.dropbox.com/scl/fo/6x6vovfkkbmjujock9px0/AJCvNGJje1RgPQUvCOoFPq0?rlkey=48lobiml61e2m1v87rr2kayh7&st=6nuet6r9&dl=1"
mkdir -p relevant_files/psychophysics_experiment
unzip relevant_files/temp_download.zip -d relevant_files/psychophysics_experiment/
rm -rf relevant_files/temp_download.zip

# If you plan on using the existing datasets and the model weights to reconstruct manuscript figures:
# Step 1: Extract the file from the relevant files folder
tar -xvf relevant_files/<weights or dataset folder>/<filename> -C relevant_files/<weights or dataset folder>
# Step 2: Navigate to the folder of interest and move it to the parent directory (using the mv command):
# a) For model weights find the model_weights folder and move it to the parent directory 
# b) For additional model weights find the additional_model_weights folder and move it to the parent directory 
# c) For datasets you can move the model-training or model-psychophysics dataset in the relevant location. However you will have to add the absolute paths in all config files provided in each subdirectory to run the notebooks
```



## Usage

### 1) Generating new contour stimuli
```bash
cd contour_integ_stimuli
```

All required functions are present in field_stim_function.py file. To generate and render a sample set of contour stimuli, run **generate_sample.ipynb**. This notebook provides a quick way to generate a sample set of contour stimuli using the default parameters.
```bash
jupyter notebook generate_sample.ipynb
```

To generate new training and psychophysics stimuli, follow these steps:  
**Step 1:** Open the **training_psychophysics_config.py file** and update the parameters and folder locations for the training and psychophysics datasets as needed.  
**Step 2:** Run the **generate_training_psychophysics.ipynb notebook**. This notebook uses the updated parameters from training_psychophysics_config.py to generate new training and psychophysics stimuli.  
```bash
jupyter notebook generate_training_psychophysics.ipynb
```


### 2) Training Contour Readout Model
```bash
cd contour_integ_models/train_model
```

All required python files are present in the contour_integ_models/lib folder. Here is a quick description:
* build_fe_ft_models.py - Contains code to build spliced models which are used to train readout models (fine-tuned or frozen) with different dnn backbones
* cutom_dataset.py - Contains code to make pytorch datasets for the contour stimuli
* feature_extractor.py - Contains code to hook and read activations from intermediate layers of a dnn backbone
* field_stim_function.py - Contains code to generate and render contour stimuli
* guided_backprop.py - Contains code to run guided backprop on readout (spliced) models
* misc_functions.py - Additional functions
* utility_functions.py - Additional utility functions
* receptive_fields.py - Comtains code to measure analytical receptive field size of units in intermediate layers of a dnn backbone


To train a contour readout model, follow these steps:  
**Step 1:** Open the **visualdiet_basemodel_config.py** file and update visual_diet_config (absolute locations and configuration of contour training stimuli) and training_config (training hyperaparameters including the dnn backbone, layer readout and finetuning mode) dictionaries.  
**Step 2:** Run the **train_contour_readout.ipynb** notebook. This notebook uses the updated parameters from visualdiet_basemodel_config.py to train models and save it in the 'contour_integ_models/saved_model' directory.  
```bash
jupyter notebook train_contour_readout.ipynb
```


### 3) Analysing performance and contour sensitivity of trained readout model
```bash
cd contour_integ_models/analyse_model
```

All required python files are present in the contour_integ_models/lib folder. Here is a quick description:
* build_fe_ft_models.py - Contains code to build spliced models which are used to train readout models (fine-tuned or frozen) with different dnn backbones
* cutom_dataset.py - Contains code to make pytorch datasets for the contour stimuli
* feature_extractor.py - Contains code to hook and read activations from intermediate layers of a dnn backbone
* field_stim_function.py - Contains code to generate and render contour stimuli
* guided_backprop.py - Contains code to run guided backprop on readout (spliced) models
* misc_functions.py - Additional functions
* utility_functions.py - Additional utility functions
* receptive_fields.py - Comtains code to measure analytical receptive field size of units in intermediate layers of a dnn backbone


To analyse a saved model, follow these steps:  
**Step 1:** Open the **visualdiet_savedmodel_config.py** file and update visual_diet_config (absolute locations and configuration of contour training stimuli) and saved_model_config (location of the saved model directory) dictionaries.  
**Step 2:** Run the **analyse_contour_readout.ipynb** notebook. This notebook uses the updated parameters from visualdiet_savedmodel_config.py to analyse saved models (including the training, validation accuracies and losses, location sensitivity, and alignment sensitivity to contours using saliency maps and guided backprop).  
```bash
jupyter notebook analyse_contour_readout.ipynb
```


### 4) Analysing human data
```bash
cd contour_integ_behavior
```

All human results from the psychophysics experiments is present in the contour_integ_behavior/contour_exp1 folder. The raw data is present in the contour_exp1/data folder. To analyse this data, run **human-psychophysics_exp1.ipynb**. This notebook loads the preprocessed data, visualizes it, removes outliers and shows the final processed human data (with the split half reliability estimates).  
```bash
jupyter notebook human-psychophysics_exp1.ipynb
```


### 5) Analysing human-model alignment of trained readout model
```bash
cd contour_integ_behavior
```

All processed human behavioral data is located in contour_integ_behavior/contour_exp1/analysis_data folder (with filename analysis.pkl).

To perform model psychophysics and compare it with humans, follow these steps:  
**Step 1:** Open the **visualdiet_savedmodel_config.py** file and update psychophysics_visual_diet_config (absolute locations and configuration of contour psychophysics stimuli) and saved_model_config (location of the saved model directory) dictionaries.  
**Step 2:** Run the **psychophysics_contour_readout.ipynb** notebook. This notebook uses the updated parameters from visualdiet_savedmodel_config.py to measurer contour signal strength in a saved model and compare models and humans at the level of individual trials and visualize the model and human performance as a function of global curvature.  
```bash
jupyter notebook human-psychophysics_contour_readout.ipynb
```


### 6) Manuscript Figures
```bash
cd manuscript_figures
```

To generate all the figures from the mansucript, follow these steps:  
**Step 1:** Complete all the steps listed in [Installation](#installation). These steps are useful to extract all the model weights and contour stimuli used in the experiements and  move the required files in the parent directory.  
**Step 2:** Open the **visualdiet_savedmodel_config.py** file and update visual_diet_config and psychophysics_visual_diet_config (absolute locations and configuration of contour psychophysics stimuli) dictionaries.  
**Step 3:**: Run the **manuscript_specific_plots.ipynb** notebook.  
```bash
jupyter notebook manuscript_specific_plots.ipynb
```



### 7) Behavioral code
Behavioral experiments were implemented using javascript (using the jspsych library - Leeuw et al., 2015), html, and css, and can be run in the browser. To get access to the behavioral code, follow these steps:
**Step 1:** Complete  the steps listed in [Installation](#installation) to extract the psychophysics experiment.
**Step 2:** The behavioral code can be found in the relevant_files folder.
```bash
cd relevant_files/psychophysics_experiment
```



### 8) Results file
Results for all the readout models is present in **manuscript_figures/results_contour_readout_models.csv**.




## Contact
If you have any questions or issues accessing the files, please contact [fenil_doshi@fas.harvard.edu](mailto:fenil_doshi@fas.harvard.edu).



## Contributors

- [Fenil R. Doshi](https://www.fenildoshi.com/)
- [Talia Konkle](https://konklab.fas.harvard.edu/)
- [George A. Alvarez](https://visionlab.harvard.edu/george/)



## Citation
In prep



## License
In prep