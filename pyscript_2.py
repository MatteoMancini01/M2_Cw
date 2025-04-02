#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
# Set seed
torch.manual_seed(23)


# In[2]:


from src.qwen import load_qwen
model_qwen, tokenizer = load_qwen()


# # Part 3 (a) (Continued..)
# 
# Plan of action:
# 
# - Preprocess the data, using `load_and_preprocess`, and split the data into `train_texts`, `val_texts` and `val_text_70` $\equiv$ 900, 100 and 100 (systems). 
# - The two validation sets `val_texts` and `val_text_70` have the same `shape` but:
#     - In `val_texts` each system has the full 100 pairs of prey and predators
#     - In `val_texts_70` each system has only the first 70 pairs of prey and predators
# - We train the model on tokenised `train_texts`
# - We validated the model by predicting the remaining 30 pair points in each of the 100 system in tokenised `val_texts_70`. 
# - We then compare the predicted results from `val_texts_70` to the gruond truth data `val_texts` (or `true_val_values` obtained with `data_scale_split`)
# - Just like for the untrained models we then want to compute MSE and RMSE 
# - And report the loss/perplexity of each trained models
# 
# We are recomended to train our model up to 10,000 steps, but we have a budgeted number of flops overall for training $10^{17}$ and due to computational power required, we are going to proceed with fewer steps first, also to familiarise with the traing procedure, before increasing the number of steps and using HPC.
# 
# In synthesis:
# 
# “We trained on 900 systems, validated on 100 full sequences for loss monitoring, and evaluated forecasting performance by generating future predictions given the first 70 steps from each validation sequence.”
# 
# All the above description has been fully prepared in `set_up_lora.py`. For flops estimation we can use `total_transformer_training_flops` in `flops.py`. The reader is invited to explore and analyse every file in `src`.

# In[3]:


# Import designed functions
from src.set_up_lora import*
from src.flops import*


# After training the model we can determine the estimate number of flops based on training steps and other metrics.
# 
# For this part we do not want to exceed 5000 steps, otherwise we will be too close to the limited number of FLOPs allowed for training
# 
# To train the model we are going to implement the function `train_lora_model` from `set_up_lora.py`.

# In[ ]:


model_lora0, loss_lora0, grad_norm_values0, loss_values0 = train_lora_model(model_qwen, tokenizer) # default steps and hyper parameters are set here


# In[ ]:


config = model_lora0.config

# Parameters
num_steps = 5000
batch_size = 4
seq_len = 512
d_model = config.hidden_size
num_heads = config.num_attention_heads
num_layers = config.num_hidden_layers
intermediate_dim = 2 * d_model  # SwiGLU
lora_rank = 4  # if using LoRA

total_flops_estimate = total_transformer_training_flops(num_steps, batch_size, seq_len, num_layers, d_model, num_heads, intermediate_dim, lora_rank)

print(f'Total number of estimated FLOPs for training LoRA with {num_steps} steps:',total_flops_estimate)


# Evaluating loss and perplexity of both tarin and validation set, there is a designed function in `set_up_lora.py`, that evaluates the perplexity and loss of the validation set, to determine the loss and perplexity of the training set, we can directly extract it from `model_lora_5000`.

# ### Loss and Perplexity 

# In[ ]:


_,val_texts, val_texts_70 = load_and_preprocess("data/lotka_volterra_data.h5")

max_steps = 5000 # CHANGE IF REQUIRED
print(f"After training with {max_steps} steps")
print(f"Training loss: {loss_lora0:.4f}")
perplexity_train = np.exp(loss_lora0)
print(f"Training perplexity: {perplexity_train:.4f}")

loss_val0, ppl_val0 = evaluate_loss_perplexity_val(model_lora0, tokenizer, val_texts, 4)
print('')
print(f'Validation loss: {loss_val0:.4f}')
print(f'Validation perplexity: {ppl_val0:.4f}')


# Saving metrics

# In[ ]:


# collecting results
collecting_results = [[loss_lora0, perplexity_train, 
                       loss_val0, ppl_val0,
                      total_flops_estimate]
                      ]
columns = ["Training Loss", "Training Perplexity", "Validation Loss", "Validation Perplexity", "FLOPs"]

# Collecting results in pd.DataFrame

best_model_results_df = pd.DataFrame(collecting_results, columns=columns)
print("Metric results from best model:")
print("")
print(best_model_results_df)

# Save results

best_model_results_df.to_csv("experiment_results/trained_lora_3a_5000/best_model_trval_loss_ppl.csv")


# ### Forecasting Missing Pair Values
# 
# After training the model, we can start using its predictive ability with the function `prediction_after_training` also defined in `set_up_lora.py`. 
# Our goal is to predict the missing 30 pairpoints in `val_texts_70`, to then compare it to the full validation set, already pre-defined in the function `prediction_after_training`. Once we have both sets we can evaluate the following metrics, error difference within each system, MSE and RMSE.

# In[ ]:


predicted_encoded0 = prediction_after_training(model_lora0, tokenizer, val_texts_70)


# ### Evaluating Metrics
# 
# To evaluate the metric mentioned above, we are going to use the designed function, `decoder_and_metrics_evaluator`, this function will return, the predicted outputs both as string-like and time=series (both outputs will be used in other functions), the true values in the validation set, and all the relevant metrics, i.e. MSE, RMSE and error in each idividual system.

# In[ ]:


predictions_decoded0, predicted_output0, true_values0, MSE_values0, RMSE_values0, error_per_system0 = decoder_and_metrics_evaluator(predicted_encoded, tokenizer)


# Saving results.

# In[ ]:


np.savez("experiment_results/trained_lora_3a_5000/predictions_decoded_trained_lora_3a.npz", *predictions_decoded0)
MSE_loaded = np.save("experiment_results/trained_lora_3a_5000/MSE_values_3a.npy", np.array(MSE_values0))
np.save('experiment_results/trained_lora_3a_5000/RMSE_values_3a', RMSE_values0)
np.savez("experiment_results/trained_lora_3a_5000/error_per_system_5000.npz", *error_per_system0)


# ### Visualisation of results
# 
# There is a designed function that wraps all the functions defined in `plotting.py` into a single function, `collective_plots`

# In[ ]:


collective_plots(predicted_encoded0, tokenizer, system_id=0, bins=30)
grad_norm_loss_plot(loss_values0, grad_norm_values0) # printing loss and grad norm over training steps


# # Part 3 (b)

# # Hyper Parameter search
# 
# In this section we are aiming to find the a set of hyper parameters, in particular "rank, learning rate" and "context lenght".

# In[ ]:


import torch
# Set seed
torch.manual_seed(23)
torch.cuda.empty_cache() #Clean gpu memory for next experiment


# In[13]:


import numpy as np
import torch
import torch.nn as nn
from src.set_up_lora import*
from src.preprocessor import*
from src.flops import*
import gc
import torch
from src.qwen import load_qwen
from src.set_up_lora import*
_, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5")
import pandas as pd


# ### Strategy:
# 
# We want to sweap through all possible combuination of the following values:
# - $r = (2,4,8)$ "rank"
# - $lr = (10^{-5}, 5 \times 10^{-5}, 10^{4})$ "learning rate"
# - $cl = 512$ "context length", i.e. fixed for now
# 
# The nested loop below will be very expensive in terms of computation, this will load Qwen2.5 nine times, if your local machine struggles to reload Qwen2.5 that many times, use the alternative code below.
# 
# After sweaping through all possible combination, we want to use the combination that provided the smallest loss/perplexity value (both will be computed within the nested loop at each itearation). Same aplies for the next HP search.

# In[14]:


_,tokenizer = load_qwen() # Load tokeniser


# In[ ]:


results_rank_lr = []

ranks = [2, 4, 8]
lrs = [1e-5, 5e-5, 1e-4]

for r in ranks:
    for lr in lrs:
        print(f"\nTraining with r={r}, lr={lr}")

        # Load fresh model
        model, _ = load_qwen()

        # Train model and compute loss/perplexity
        trained_model, final_loss, _, _ = train_lora_model(model, tokenizer, lora_rank=r, learning_rate=lr, train_steps=1000)
        ppl_train = np.exp(final_loss)

        # Compute validation loss and perplexity
        val_loss, _ = evaluate_loss_perplexity_val(trained_model, tokenizer, val_texts, 4)
        ppl_val = np.exp(val_loss)
        
        # Extracting config information to determine estimate number of flops
        config = trained_model.config
        d_model = config.hidden_size
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers
        intermediate_dim = 2 * d_model  # SwiGLU

        # Compute total estimate of flops
        total_flops_estimate = total_transformer_training_flops(1000, 4, 512, num_layers, d_model, num_heads, intermediate_dim, lora_rank=r)

        # Collecting results
        results_rank_lr.append({"rank": r, "learning_rate": lr, "Train Loss": final_loss, "Train Perplexity": ppl_train,
                                "Validation Loss": val_loss, "Validation Perplexity":ppl_val,"Estimated Flops": total_flops_estimate})
        print(f"-> Train Loss: {final_loss:.4f}, Perplexity: {ppl_train:.2f}")
        print(f"-> Validation Loss: {val_loss:.4f}, Perplexity: {ppl_val:.2f}")
        print(f"-> Estimated Flops: {total_flops_estimate}")

        # Clean up to free GPU memory
        del model
        del trained_model
        torch.cuda.empty_cache()
        gc.collect()


# In[ ]:


# Saving results as a csv file
HP_search_rlr_df = pd.DataFrame(results_rank_lr)
print(HP_search_rlr_df)
HP_search_rlr_df.to_csv("experiment_results/hp_tuning_results/hp_tun_rank_lr.csv")


# After determining best hyper parameters for "rank" and "learning rate", we can procede to determine which of the three context lengths $[128, 512, 768]$ perform the best for a maximun of 2000 RLPPP steps

# In[17]:


import torch
# Set seed
torch.manual_seed(23)
torch.cuda.empty_cache() #Clean gpu memory for next experiment


# In[ ]:


results_cl = []
context_lengths = [128, 512, 768]
best_r = 8
best_lr = 1e-4

for cl in context_lengths:
    print(f"\nTraining with context_lenghts = {cl}")

    # Load fresh model
    model, _ = load_qwen()
    # Train the model and compute loss
    trained_model, final_loss, _, _ = train_lora_model(model, tokenizer, lora_rank=best_r, learning_rate=best_lr, max_ctx_length=cl, train_steps=1000)
    ppl_train = np.exp(final_loss)


    # Computing validation loss and perplexity
    val_loss, _ = evaluate_loss_perplexity_val(trained_model, tokenizer, val_texts, 4, max_length=cl)
    ppl_val = np.exp(val_loss)
    
    # Extracting config info from the model to estimate flops
    config = trained_model.config
    d_model = config.hidden_size
    num_heads = config.num_attention_heads
    num_layers = config.num_hidden_layers
    intermediate_dim = 2 * d_model  # SwiGLU

    #  Computing total estimate of flops
    total_flops_estimate = total_transformer_training_flops(1000, 4, cl, num_layers, d_model, num_heads, intermediate_dim, lora_rank=best_r)

    # Collecting results
    results_cl.append({"context_lengths": cl, "Train Loss": final_loss, "Train Perplexity": ppl_train,
                       "Validation Loss": val_loss, "Validation Perplexity": ppl_val,"Estimated Flops": total_flops_estimate})
    
    # Print results at each stage
    print(f"-> Train Loss: {final_loss:.4f}, Perplexity: {ppl_train:.2f}")
    print(f"-> Validation Loss: {val_loss:.4f}, Perplexity: {ppl_val:.2f}")
    print(f"-> Estimated Flops: {total_flops_estimate}")

    # Clean up to free GPU memory
    del model
    del trained_model
    torch.cuda.empty_cache()
    gc.collect()


# In[ ]:


# Saving results as a csv file
HP_search_cl_df = pd.DataFrame(results_cl)
print(HP_search_cl_df)

HP_search_cl_df.to_csv("experiment_results/hp_tuning_results/hp_tun_cl.csv")


# # Part 3 (c)

# In this section we will pefrom full training and validation using the best hyper-parameters determined in the previus section, determine the number of flops and all the metrics just like in previus parts, and compare resuts of the tuned model to the trained but not tuned Qwen2.5 with LoRA.

# In[1]:


import torch
# Set seed
torch.manual_seed(23)
torch.cuda.empty_cache() #Clean gpu memory for next experiment


# In[2]:


# Follecting from previus results

r_opt = 8 # Optimal Rank
lr_opt = 1e-4 # optimal Learning Rate
cl_opt = 512 # Optimal Context Length


# Training Model with the above parameters (and determine loss/perplexity)

# In[3]:


from src.qwen import load_qwen
model_opt, tokenizer = load_qwen() # Loading Qwen2.5


# In[ ]:


from src.set_up_lora import*
model_opt_train1, loss_train1, grad_norm_values1, loss_values1 = train_lora_model(model_opt, tokenizer, lora_rank=r_opt, learning_rate=lr_opt, batch_size=4, max_ctx_length=cl_opt,train_steps=5000) # Training model


# Collecting train and validation loss/perplexity and FLOPs

# In[ ]:


from src.flops import*

# Computing validation loss and perplexity

_, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5") # Reloading validation set

ppl_best_train = np.exp(loss_train1) # Perplexity on trained model

# Validation loss and perplexity of best model
loss_best_val, ppl_best_val = evaluate_loss_perplexity_val(model_opt_train1, tokenizer, val_texts, 4,  max_length=cl_opt)

# Extracting config info from the model to estimate flops
config = model_opt_train1.config
d_model = config.hidden_size
num_heads = config.num_attention_heads
num_layers = config.num_hidden_layers
intermediate_dim = 2*d_model  # SwiGLU

#  Computing total estimate of flops
total_flops_for_best_model = total_transformer_training_flops(5000, 4, cl_opt, num_layers, d_model, num_heads, intermediate_dim, lora_rank=r_opt)

# collecting results
collecting_results = [[loss_train1, ppl_best_train, 
                       loss_best_val, ppl_best_val,
                      total_flops_for_best_model]
                      ]
columns = ["Training Loss", "Training Perplexity", "Validation Loss", "Validation Perplexity", "FLOPs"]

# Collecting results in pd.DataFrame

best_model_results_df = pd.DataFrame(collecting_results, columns=columns)
print("Metric results from best model:")
print("")
print(best_model_results_df)

# Save results

best_model_results_df.to_csv("best_model_results/best_model_trval_loss_ppl.csv")


# Using best model for predictions

# In[ ]:


# Prediction after training
_,_,val_texts_70 = load_and_preprocess("data/lotka_volterra_data.h5") # Reload 70% of validation set

# Compute prediction
predicted_encoded1 = prediction_after_training(model_opt_train1, tokenizer, val_texts_70)


# Determining metrics from predictions, MSE, RMSE and error in each system

# In[ ]:


predictions_decoded1, predicted_output1, true_values1, MSE_values1, RMSE_values1, error_per_system1 = decoder_and_metrics_evaluator(predicted_encoded1, tokenizer)


# Saving metrics

# In[ ]:


np.savez("experiment_results/best_model_results/predictions_decoded_best.npz", *predictions_decoded1)
MSE_loaded = np.save("experiment_results/best_model_results/MSE_values_best.npy", np.array(MSE_values1))
np.save('experiment_results/best_model_results/RMSE_values_best', RMSE_values1)
np.savez("experiment_results/best_model_results/error_per_system_best.npz", *error_per_system1)


# ### Visualisation

# In[ ]:


collective_plots(predicted_encoded1, tokenizer)
grad_norm_loss_plot(loss_values1, grad_norm_values1) # printing loss and grad norm over training steps


# # 15000 Steps Experiment

# In[2]:


import numpy as np
import torch
import torch.nn as nn
from src.set_up_lora import*
from src.preprocessor import*
from src.flops import*
import gc
import torch
from src.qwen import load_qwen
from src.set_up_lora import*
_, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5")
import pandas as pd


# In[3]:


# Set seed
torch.manual_seed(24)
torch.cuda.empty_cache() #Clean gpu memory for next experiment


# In[4]:


# Follecting from previus results

r_opt = 8 # Optimal Rank
lr_opt = 1e-4 # optimal Learning Rate
cl_opt = 512 # Optimal Context Length


# In[5]:


model_opt, tokenizer = load_qwen() # Loading Qwen2.5


# In[ ]:


model_opt_train2, loss_train2, grad_norm_values2, loss_values2 = train_lora_model(model_opt, tokenizer, lora_rank=r_opt, learning_rate=lr_opt, batch_size=4, max_ctx_length=cl_opt,train_steps=15000) # Training model


# In[ ]:


# Computing validation loss and perplexity

_, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5") # Reloading validation set

ppl_best_train = np.exp(loss_train2) # Perplexity on trained model

# Validation loss and perplexity of best model
loss_best_val, ppl_best_val = evaluate_loss_perplexity_val(model_opt_train2, tokenizer, val_texts, 4,  max_length=cl_opt)

# Extracting config info from the model to estimate flops
config = model_opt_train2.config
d_model = config.hidden_size
num_heads = config.num_attention_heads
num_layers = config.num_hidden_layers
intermediate_dim = 2*d_model  # SwiGLU

#  Computing total estimate of flops
total_flops_for_best_model = total_transformer_training_flops(15000, 4, cl_opt, num_layers, d_model, num_heads, intermediate_dim, lora_rank=r_opt)

# collecting results
collecting_results = [[loss_train2, ppl_best_train, 
                       loss_best_val, ppl_best_val,
                      total_flops_for_best_model]
                      ]
columns = ["Training Loss", "Training Perplexity", "Validation Loss", "Validation Perplexity", "FLOPs"]

# Collecting results in pd.DataFrame

best_model_results_df = pd.DataFrame(collecting_results, columns=columns)
print("Metric results from best model:")
print("")
print(best_model_results_df)

# Save results

best_model_results_df.to_csv("experiment_results/best_model_results/best_model_trval_loss_ppl.csv")


# In[ ]:


# Prediction after training
_,_,val_texts_70 = load_and_preprocess("data/lotka_volterra_data.h5") # Reload 70% of validation set

# Compute prediction
predicted_encoded2 = prediction_after_training(model_opt_train2, tokenizer, val_texts_70)


# In[ ]:


predictions_decoded2, predicted_output2, true_values2, MSE_values2, RMSE_values2, error_per_system2 = decoder_and_metrics_evaluator(predicted_encoded2, tokenizer)


# In[ ]:


np.savez("experiment_results/raw_performance_experiment_results/predictions_decoded_best.npz", *predictions_decoded2)
MSE_loaded = np.save("experiment_results/raw_performance_experiment_results/MSE_values_best.npy", np.array(MSE_values2))
np.save('experiment_results/raw_performance_experiment_results/RMSE_values_best', RMSE_values2)
np.savez("experiment_results/raw_performance_experiment_results/error_per_system_best.npz", *error_per_system2)


# In[ ]:


collective_plots(predicted_encoded2, tokenizer)
grad_norm_loss_plot(loss_values2, grad_norm_values2) # printing loss and grad norm over training steps

