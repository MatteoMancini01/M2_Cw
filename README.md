# M2_Cw
Coursework on Deep Learning for Data Intensive Science :robot::brain:
----------------------------------------------------

# Acknowledgment

I would like to thank my brother Riccardo for running my codes on his computer. I found myself in a tricky situation, this project required a lot of computing power and memory. The machine I use is a Lenovo Legion 5 15ACH56H, RAM 16GB (of which 13.9GB usable), Processor AMD Ryzen 7 58000H with Radeon Graphics 3.20 GHz, NVIDIA GeForce RTX 3070 Laptop GPU: 8GB GDDR6, boost clock 1560 MHz, maximum graphics power 130W.
I have quite a remarkable machine, but unfortunately with not enough memory available to withstand the tasks execution of this project. 
In comparison Riccardo’s PC specs are, CPU: AMD Ryzen 9 7900X3D, RAM 32GB, GPU: GeForce RTX 4080 Super 16GB.

Machines comparison:

|Spec|My Laptop|Riccardo's PC|
|----|---------|-------------|
|CPU|AMD Ryzen 7 58000H|AMD Ryzen 9 7900X3D|
|RAM| 16GB|32GB|
|GPU|NVIDIA GeForce RTX 3070 Laptop GPU: 8GB GDDR6|GeForce RTX 4080 Super 16GB|


The two machines are quite far in terms of memory and computational power. I have tried to train Qwen2.5-0.5B injected with LoRA layers for 5000 steps, this procedure took approximately 8.5 hours, the same experiment was conducted with Riccardo’s PC and completed in approximately 20 minutes. 
The reader is welcome to clone the repository and play with the codes, but beware, you need a high-level performance machine, with a minimum of RAM 32GB.  GPUs may help speeding up the training process, however between experiments it is recommended to empty GPUs cache. 


I would like to acknowledge the invaluable assistance of ChatGPT-4 and Grammarly in the preparation of this project. The tools played a significant role in:

- Debugging and resolving coding challenges encountered during the development of the Python package.
- Providing insights and suggestions for improving the structure and functionality of the code.
- Generating concise and accurate summaries of complex texts to enhance understanding and clarity.

While ChatGPT-4 contributed significantly to streamlining the development process and improving the quality of outputs, all results were rigorously reviewed, tested, and refined to ensure their accuracy, relevance, and alignment with project objectives.
----------------------------------------------------
# LoRA Fine-Tuning for Time Series with Qwen2.5-0.5B
----------------------------------------------------
## Overview :compass:
This project aims to investigate the forecasting ability of large language models [(LLMs)](https://en.wikipedia.org/wiki/Large_language_model) on time series data. What takes us here, is the fact that traditional time series forecasting methods often struggle to capture long-term dependencies or patterns in time series data. The approach we just suggested, i.e. using LLMs to model time series data is commonly referred to as [LLMTIME](https://arxiv.org/abs/2310.01728), where time series are treated as textual sequences for language modelling tasks. This is exactly what we have adopted, allowing seamless integration with language models. Using the pre-trained [Qwen2.5-0.5B](https://github.com/QwenLM/Qwen2.5)-Instruct model from [Hugging Face](https://huggingface.co/Qwen), we fine-tune it for forecasting task via parameter-efficient Low Rank Adaptation (LoRA), enabling better predictions compared to the pure forecasting ability of Qwen2.5-0.5B. Furthermore, for the nature of this project, i.e. heavy machinery computation, we are instructed to track floating point operations per seconds (FLOPs) during training up to order $10^{17}$. 



---------------------------------------------------
## Getting started :rocket:
---------------------------------------------------
### Python Virtual Enviroments :snake::test_tube:

1. Create a Virtual Environment

   Run the following command to create your virtual environment

   ``` bash
    python -m venv <your_env>

- If the above command fails, please try:
   ```bash
   python3 -m venv <your_env>

Replace `<your_env>` with your preferred environment name, e.g. `m2_venv`.

2. Activate your virtual environment

  Activate your virtual environment with:
   ```bash
    source <your_env>/bin/activate
   ```
  Deactivate your environment with:
   ```bash
    deactivate
   ```
3. To install all the required libraries please run the command:
   ```bash
   pip install -r requirements.txt
   ```
---------------------------------------------------
### Conda Environments :snake::test_tube:

1. Create a Conda Environment
   Run the following command to create your Conda environment

    ```bash
    conda env create -f environment.yml
    ```

All the required libraries will be automatically installed.

2. Activate your Conda envorpnment

    ```bash
    conda activate m2_venv
    ```

   To deactivate: 
   
    ```bash
    conda deactivate
    ```

---------------------------------------------------
## Key Features :key:

:memo: List of all the things the pipeline and codebase do:

- Test Qwen2.5-0.5B model's forecasting ability on tokenised time series :white_check_mark:
- Estimate FLOPs during training :white_check_mark:
- Fine-tunes Qwen2.5-0.5B using LoRA layers on tokenised time series :white_check_mark:
- Efficiient training with Huggug Face [`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator) :white_check_mark:
- Implements [gradient clipping](https://huggingface.co/docs/accelerate/v0.2.0/quicktour.html) :white_check_mark: 
- Tracks and visualises training loss and gradient norms (L2) during and after training :white_check_mark: 
- Includes Hyper-Parameters search hooks :white_check_mark: 
---------------------------------------------------

## :open_file_folder: Directory Structure 
<pre>
📂 M2_Cw/
│
├── 📂 data/
│   └── lotka_volterra_data.h5                    # Where the data is stored
│
├── 📂 experiment_results/                       # Where results are stored 
│   ├── 📁 best_model_results
│   ├── 📁 hp_tuning_results
│   ├── 📁 tuned_lora_3c_hp_5000
│   ├── 📁 trained_lora_3a_5000
│   ├── 📁 untrained_qwen_plus_lora_results
│   └── 📁 untrained_qwen_results
│
├── 📂 instructions/                            # Instructions and  reading material
│   ├── 📄 llmtime.pdf
│   ├── 📄 main.pdf
│   └── 📄 qwen.pdf
│
├── 📂 plots/                                  # Where plots are stored
│   ├── 📁 trained_lora_3a_5000
│   ├── 📁 tuned_lora_15000
│   ├── 📁 tuned_lora_5000
│   ├── 📁 untrained_lora
│   └── 📁 untrained_qwen
│
├── 📂 src/                                   # Where all the required functions are stored
│   ├── 🐍 flops.py
│   ├── 🐍 lora_skeleton.py
│   ├── 🐍 minimal_lora_eval.py
│   ├── 🐍 plotting.py
│   ├── 🐍 preprocessor.py
│   ├── 🐍 qwen.py
│   └── 🐍 set_up_lora.py
│
├── 📄 .gitignore
├── 📓 Baseline.ipynb                       # Where Qwen2.5-0.5B model's forecasting is evaluated
├── ⚖️  LICENSE
├── 📓 Qwen_LoRA_untrained.ipynb            # Qwen + LoRA layers evaluation (untrained)
├── 📘 README.md
├── ⚙️ environment.yml                      # Conda SetUp
├── 📄 ProjectReport.pdf
├── 🐍 pyscript_0.py                        # Baseline.ipynb converted to Python Script
├── 🐍 pyscript_1.py                        # Qwen_LoRA_untrained.ipynb converted to Python Script
├── 🐍 pyscript_2.py                        # train_tune_LoRA.ipynb converted to Python Script
├── ⚙️  requirement.txt                     # Python Environment SetUp
└── 📓 train_tune_LoRA.ipynb
</pre>
---------------------------------------------------
## :bulb:Understanding each :page_facing_up:File \& :file_folder:Directory 

<pre>
├── 📂 data/
│   └── lotka_volterra_data.h5    
</pre>
The directory `data/` contains a single file, lotka_volterra_data.h5. This is saved as a [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format),  a hierarchical data container optimised for large-scale numerical arrays. Its raw structure is a 3D array of shape (N, T, 2), where N (=1000) is the number of systems, T (=100) is the number of timesteps, and the last dimension corresponds to the prey and predator populations, respectively. To unpack and upload the dataset we have used the package :package:[`h5py`](https://docs.h5py.org/en/stable/quick.html).

---------------------------------------------------
<pre>
├── 📂 experiment_results/                     
│   ├── 📁 best_model_results
│   ├── 📁 hp_tuning_results
│   ├── 📁 tuned_lora_3c_hp_5000
│   ├── 📁 trained_lora_3a_5000
│   ├── 📁 untrained_qwen_plus_lora_results
│   └── 📁 untrained_qwen_results
</pre>
The directory experiment_results contains other directories that stores other sub-directories, where we stored our experiment and training results, the reader is welcome to explore them or load the results directly rather than training models for hours.

---------------------------------------------------
<pre>
├── 📂 instructions/         
│   ├── 📄 llmtime.pdf
│   ├── 📄 main.pdf
│   └── 📄 qwen.pdf
</pre>
This directory contains three .pdf files. 

|File Name|Description|
|---------|-----------|
|llmtime.pdf|This paper proposes reframing time series forecasting as a language modeling task, allowing large language models (LLMs) like Qwen2.5 to be trained on time series data using next-token prediction.|
|qwen.pdf| This paper introduces Qwen2, a family of open-source large language models ranging from 0.5B to 72B parameters, trained by Alibaba Cloud. It outlines improvements in training strategy, architecture, and performance.|
|main.pdf|This file contains the instructions followed for this project.|

---------------------------------------------------

<pre>
├── 📂 plots/                                
│   ├── 📁 trained_lora_3a_5000
│   ├── 📁 tuned_lora_3c_5000
│   ├── 📁 tuned_lora_5000
│   ├── 📁 untrained_lora
│   └── 📁 untrained_qwen

</pre>
The directory `plots/` contains other sub-directories that contains meaningful plots, e.g. metric plots, true vs predicted values plots etc.

---------------------------------------------------
<pre>
├── 📂 src/                                   # Where all the required functions are stored
│   ├── 🐍 flops.py
│   ├── 🐍 lora_skeleton.py
│   ├── 🐍 minimal_lora_eval.py
│   ├── 🐍 plotting.py
│   ├── 🐍 preprocessor.py
│   ├── 🐍 qwen.py
│   └── 🐍 set_up_lora.py
</pre>
The directory `src/` (software projects), contains all the main application codes, we use it to separate core functionality from supporting files such as configs, data, notebooks, or documentation. The table below illustrates the general purposes of each `*.py` file in `src/`, with the name of each respective function or class, the reader is invited to explore those files as each function and class is well documented with docstrings, explaining function’s purpose, inputs, outputs and examples. 

|Python File|Functions and/or Classes|General Purpose|
|-----------|---------|---------------|
|[`flops.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/src/flops.py)|`total_transformer_training_flops()` (function), `compute_generation_flops()` (function)| The purpose of this file is to store functions designed for estimating the number of FLOPs used while training, or (beyond the purpose of this project) estimate the number of FLOPs used during predictions. Flops.py is used in the following files, [`Baseline.ipynb`](https://github.com/MatteoMancini01/M2_Cw/blob/main/Baseline.ipynb), [`train_tune_LoRA.ipynb`](https://github.com/MatteoMancini01/M2_Cw/blob/main/train_tune_LoRA.ipynb), and their respective converted to Python files [`pyscript_0.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/pyscript_0.py) and [`pyscript_2.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/pyscript_2.py).|
|[`lora_skeleton.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/src/lora_skeleton.py)| `LoRALinear` (class), `process_sequence()`(function)|This Python file was provided to us, aiming to illustrate and give us the basics to build our own training procedure, models and preprocessing/processing data. This file is not used anywhere!|
|[`minimal_lora_eval.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/src/minimal_lora_eval.py)|`LoRALinear` (class), `process_sequence()`(function) and `evaluate_loss()` (function)|This Python file is very similar to `lora_skeleton.py`, sharing the same class and one of the functions. This was designed to test the forecasting ability of Qwen2.5-0.5B with LoRA layers, without any training. `minimal_lora_eval.py` is only used in [`Qwen_LoRA_untrained.ipynb`](https://github.com/MatteoMancini01/M2_Cw/blob/main/Qwen_LoRA_untrained.ipynb) along with its converted Python script copy [`pyscript_1.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/pyscript_1.py).|
|[`plotting.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/src/plotting.py)| `PlotProject` (class) containing the following functions: `plot_pred_vs_true()`, `plot_error_hist_system()`, `plot_hist_MSE()` and `plot_hist_RMSE()`, outside that class the remaining functions; `pred_vs_true_visualisation()`, `grad_norm_loss_plot()`|As the name suggests, this file contains all the necessary functions and class to visualise and interpret results whether they come from model predictions with or without training. This file is imported and widely used in all Jupyter Notebooks and their respective converted Python copies, in this repository.|
|[`preprocessor.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/src/preprocessor.py)| `Preprocessor` a class containing the following functions; `scaling_operator()`, `array_to_string()`and `sting_to_array()`. Outside `Preprocessor` the following functions `load_and_preprocess()`, `data_scale_split()` and `sequence_length_array()`.|Each function in `preprocessor.py()` is designed for a specific purpose, in synthesis, as the name says, here we store all the functions that are used to load and preprocess the data, as well as calculating the maximum number of tokens after tokenisation.|
|[`qwen.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/src/qwen.py)|Contains two functions; `load_qwen()` and `tokenizer()`|This might be a oversimplification, but this file stores two functions designed to load Qwen2.5-0.5B model and tokenizer, and tokenise string like data. Again this function is used in every Jupyter Notebook and Python files.|
|[`set_up_lora.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/src/set_up_lora.py)|Functions: `process_sequences()`, `evaluate_loss()`, `train_lora_model()`, `evaluate_loss_perplexity_val()`, `prediction_after_training()`, `decoder_and_metrics_evaluator()`, `collective_plots()`. Class `LoRALinear`|This Python files hosts all the functions utilised for, training, evaluating and visualise results. This is used only in [`train_tune_LoRA.ipynb`](https://github.com/MatteoMancini01/M2_Cw/blob/main/train_tune_LoRA.ipynb)Jupyter Notebook and its copied (converted equivalent) Python file [`pyscript_2.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/pyscript_2.py).|


---------------------------------------------------
### Jupyter Notebooks📓 and Python Scripts:snake:

<pre>
├── 📓 Baseline.ipynb                       # Where Qwen2.5-0.5B model's forecasting is evaluated
├── 📓 Qwen_LoRA_untrained.ipynb            # Qwen + LoRA layers evaluation (untrained)  
├── 📓 train_tune_LoRA.ipynb                # Where training and hp search occurs
├── 🐍 pyscript_0.py                        # Baseline.ipynb converted to Python Script
├── 🐍 pyscript_1.py                        # Qwen_LoRA_untrained.ipynb converted to Python Script                
└── 🐍 pyscript_2.py                        # train_tune_LoRA.ipynb converted to Python Script
</pre>
Before we describe what each file contains and used for, I would like to state that all results are set to be reproducible, using  `torch.manual_seed()`, however these results are machine dependent, thus if any user run my notebooks or Python scripts, results are not guaranteed to be the same as my findings (see PyTorch [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) Documentation). 

The table below will provide an overview of what tasks each file covers:

|Jupyter Notebook \& Python Copy| Tasks|
|-------------------------------|------|
|:notebook: [`Baseline.ipynb`](https://github.com/MatteoMancini01/M2_Cw/blob/main/Baseline.ipynb), :snake: [`pyscript_0.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/pyscript_0.py)| Here is where we execute all the tasks required for 2. Baseline see [instructions/main.pdf](https://github.com/MatteoMancini01/M2_Cw/blob/main/instructions/main.pdf), this includes implementing LLMTIME preprocessing scheme for the time series data, providing two examples of the pre-processed and tokenised results. Evaluating the untrained Qwen2.5-Instruct model’s forecasting ability and metric evaluation (MSE, RMSE, error in each system).|
|:notebook: [`Qwen_LoRA_untrained.ipynb`](https://github.com/MatteoMancini01/M2_Cw/blob/main/Qwen_LoRA_untrained.ipynb), :snake: [`pyscript_1.py`](https://github.com/MatteoMancini01/M2_Cw/blob/main/pyscript_1.py)|Here is were we covered the first part of task 3(a), see [instructions/main.pdf](https://github.com/MatteoMancini01/M2_Cw/blob/main/instructions/main.pdf), i.e. adopting LoRA implementation to Qwen2.5-Instruct model, by wrapping the query and value projection layers with `LoRALinear` (class)  layers. Surprisingly this is one of the largest notebooks despite only covering a small section of the project. An additional test we provide in this notebook, that is not a required task, is to test Qwen2.5 with LoRA layers without training.|
|:notebook:[`train_tune_LoRA.ipynb`](https://github.com/MatteoMancini01/M2_Cw/blob/main/train_tune_LoRA.ipynb), :snake:[pyscript_2.py](https://github.com/MatteoMancini01/M2_Cw/blob/main/pyscript_2.py)|This is where the “magic” happens, we executed the remaining tasks from [instructions/main.pdf](https://github.com/MatteoMancini01/M2_Cw/blob/main/instructions/main.pdf), i.e. Fine-Tune Qwen2.5 with LoRA layers, Hyper Parameter search, and finally conduce final training experiment based on the results of our Hyper Parameter search, while keeping track of all the FLOPs used in training. Of course, to evaluate and compare results we also determined the respective metrics. Instead of running a big final experiment we decided to split it into two smaller final experiments, as a pair of HP search results seem promising, both with no overfitting and gradient norm seemed well behave.|

---------------------------------------------------

### Remaining Files

<pre>

├── 📄 .gitignore                      
├── ⚖️  LICENSE           
├── 🗺️ README.md
├── ⚙️ environment.yml                     # Conda SetUp
├── 📄 ProjectReport.pdf
└── ⚙️  requirement.txt                    # Python Environment SetUp
</pre>

1. :page_facing_up:`.gitignore`: Is the file that tells Git which files or folders to ignore, preventing them from being tracked or committed.
2. :balance_scale:`LICENSE`:  Defines how others can use, modify, and distribute your code, protecting both the creator and users legally. 
3. :book:`README.md`: You are here! 🗺️
4. :gear:[`environment.yml`](https://github.com/MatteoMancini01/M2_Cw/blob/main/environment.yml): Helps you set up a Conda Environment by downloading all the required packages :package:.
5. :page_facing_up:`ProjectReport.pdf`: Report containing all the findings.
6. :gear:[`requirement.txt`](https://github.com/MatteoMancini01/M2_Cw/blob/main/requirements.txt): Helps you set up a Python Environment by downloading all the required packages :package:. 

---------------------------------------------------

# :books:References \& :link:Links

1. :books:[Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820) - Nate Gruver, Marc Finzi, Shikai Qiu \& Andrew Gordon Wilson et al. 2024
2. :books:[Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) -  An Yang, Baosong Yang, Beichen Zhang \& others et al. 2025
3. :link:[Qwen2.5-0.5B](https://github.com/QwenLM/Qwen2.5)
4. :link:[Hugging Face](https://huggingface.co/Qwen)