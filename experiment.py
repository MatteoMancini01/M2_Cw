#%%
from src.plotting import*
#%%
# Load error computed between true and predicted pairs prey and predator, npz file
loaded_error = np.load("untrained_qwen_results/error_per_system.npz")
error_per_system_loaded = [loaded_error[key] for key in loaded_error]
#%%
for i in range(2):
    PlotProject.plot_error_hist_system(error_per_system_loaded, system_id=i, bins=30)
#%%
loaded = np.load("untrained_qwen_plus_lora_results/my_decoded_predictions.npz")
my_decoded_predictions = [loaded[key] for key in loaded]
#%% 
from src.preprocessor import*

train_traj, val_traj, _ = load_and_preprocess('data/lotka_volterra_data.h5')
#%%
my_decoded_predictions
#%%
val_traj
PlotProject.plot_pred_vs_true(my_decoded_predictions, val_traj, 4)