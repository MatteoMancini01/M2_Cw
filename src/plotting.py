import matplotlib.pyplot as plt
import numpy as np
from src.preprocessor import Preprocessor

string_to_array = Preprocessor.string_to_array

class PlotProject():

    ''' 
    PlotProject is a class designed to host all the plotting codes required for this project
    '''

    def plot_pred_vs_true(pred_arr, true_arr_str, system_id):

        """
        Plots the predicted versus true prey-predator population dynamics over time.

        This function takes a predicted trajectory array, a string representation of 
        the true trajectory data, and a system identifier to visualize the prey-predator 
        population trends over a specified time range.

        Parameters:
        -----------
        pred_arr : list of numpy.ndarray
            A list of predicted trajectory arrays where each entry corresponds to a 
            specific system. Each array should have two columns: one for prey and 
            another for predator populations.
        
        true_arr_str : list of str
            A list of string representations of true trajectory data, which are 
            converted back into numerical arrays. Each entry represents a system's 
            population dynamics.
        
        system_id : int
            The index of the system to visualize, determining which trajectory 
            from `pred_arr` and `true_arr_str` to use.

        Functionality:
        --------------
        - Converts the string representation of the true trajectory back into a numerical array.
        - Extracts the predicted and true data for the specified system.
        - Ensures both arrays are trimmed to the same length to avoid shape mismatches.
        - Plots the original and predicted prey and predator populations over a defined time range.

        Notes:
        ------
        - The function assumes that the first column of the trajectory arrays corresponds 
        to prey population and the second column corresponds to predator population.
        - Predictions may contain a different number of data points compared to the true data.
        - The visualization starts from index 50 onwards to allow for better comparison.

        Outputs:
        --------
        - A matplotlib plot showing prey and predator populations over time with 
        dashed lines representing the true data and solid lines representing the 
        predicted data.

        Example Usage:
        --------------
        >>> plot_pred_vs_true(pred_arr, true_arr_str, system_id=3)
        """


        index = system_id 
        traj_00 = string_to_array(true_arr_str[index]) 
        dec_00 = pred_arr[index] 
        print('Decoded prediction shape:',dec_00.shape) 
        print('True dataset shape:',traj_00.shape)
        
        traj_0 = traj_00[70:]
        dec_0 = dec_00[70:]
        min_length = min(len(traj_0), len(dec_0))

        
        time_step = np.linspace(140, 200, min_length)  # Adjust time range
        prey_values = traj_0[:min_length, 0]  # Trim original prey
        predator_values = traj_0[:min_length, 1]  # Trim original predator

        predict_prey = dec_0[:min_length, 0]  # Trim predicted prey
        predict_predator = dec_0[:min_length, 1]  # Trim predicted predator

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(time_step, prey_values, label="Original Prey", color="blue", linestyle="dashed")
        plt.plot(time_step, predator_values, label="Original Predator", color="red", linestyle="dashed")

        plt.plot(time_step, predict_prey, label="Predicted Prey", color="blue")
        plt.plot(time_step, predict_predator, label="Predicted Predator", color="red")

        # Labels and Title
        plt.xlabel("Time Steps")
        plt.ylabel("Population")
        plt.title(f"Prey-Predator Dynamics Over Time, System_ID: {index+900}")
        plt.legend()

        # Show the plot
        plt.show()


    
    def plot_error_hist_system(error_per_system, system_id, bins):
        """
        Plots separate histograms for signed errors of prey and predator predictions for a specific system.

        This function visualizes the error distribution for a given system by plotting two side-by-side
        histograms: one for prey errors and one for predator errors. It also marks the mean error with
        a red dashed line.

        Parameters:
        - error_per_system (dict): A dictionary where keys are system IDs and values are tuples:
            - error_per_system[system_id][0]: List or np.array of errors for prey (y_true_prey - y_pred_prey).
            - error_per_system[system_id][1]: List or np.array of errors for predator (y_true_predator - y_pred_predator).
        - system_id (int): The ID of the system to visualize.
        - bins (int): The number of bins to use in the histograms.

        Returns:
        - None: Displays the histograms but does not return any values.

        The function performs the following:
        - Extracts the prey and predator errors for the given system.
        - Creates two side-by-side histograms:
            1. Prey Error Histogram: Shows the distribution of errors in prey predictions.
            2. Predator Error Histogram: Shows the distribution of errors in predator predictions.
        - Adds a vertical dashed red line to indicate the mean error in each histogram.

        Example:
        --------
        >>> import numpy as np

        >>> # Example error data for multiple systems
        >>> error_per_system = {
        >>>     0: (np.array([0.2, -0.5, 0.1]), np.array([-0.3, 0.4, -0.1])),
        >>>     1: (np.array([0.5, -0.7, 0.2]), np.array([-0.6, 0.3, 0.1]))
        >>> }

        >>> # Plot error histograms for System 0 with 10 bins
        >>> plot_error_hist_system(error_per_system, system_id=0, bins=10)
        """
        prey_errors = error_per_system[system_id][0]
        predator_errors = error_per_system[system_id][1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Prey Error Histogram
        axes[0].hist(prey_errors, bins=bins, edgecolor="black", alpha=0.7)
        axes[0].set_title(f"Prey Error Histogram (System {system_id+900})")
        axes[0].set_xlabel("Error (y_true - y_pred)")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(x=np.mean(prey_errors), color="red", linestyle="dashed", label=f"Mean Error: {np.mean(prey_errors):.2f}")
        axes[0].legend()

        # Predator Error Histogram
        axes[1].hist(predator_errors, bins=bins, edgecolor="black", alpha=0.7)
        axes[1].set_title(f"Predator Error Histogram (System {system_id+900})")
        axes[1].set_xlabel("Error (y_true - y_pred)")
        axes[1].axvline(x=np.mean(predator_errors), color="red", linestyle="dashed", label=f"Mean Error: {np.mean(predator_errors):.2f}")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


    def plot_hist_MSE(MSE_df, bins):
        """
        Plots separate histograms for the Mean Squared Error (MSE) of prey and predator predictions.

        This function takes a DataFrame containing MSE values for prey and predator predictions
        across multiple systems and visualizes their distributions using histograms.

        Parameters:
        - MSE_df (pd.DataFrame): A DataFrame containing at least two columns:
            - 'MSE for prey': MSE values for prey predictions.
            - 'MSE for predator': MSE values for predator predictions.
        - bins (int): The number of bins to use in the histograms.

        Returns:
        - None: Displays the histograms but does not return any values.

        The function creates two side-by-side histograms:
        1. The first histogram shows the distribution of MSE values for prey predictions.
        2. The second histogram shows the distribution of MSE values for predator predictions.

        Example:
        --------
        >>> import pandas as pd
        
        >>> # Example MSE data
        >>> data = {'MSE for prey': [0.2, 0.5, 0.7, 0.1], 'MSE for predator': [0.3, 0.6, 0.8, 0.2]}
        >>> mse_df = pd.DataFrame(data)

        >>> # Plot histograms with 10 bins
        >>> plot_hist_MSE(mse_df, bins=10)
        """
        prey_mse = MSE_df['MSE for prey'].tolist()
        predator_mse = MSE_df['MSE for predator'].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Prey Error Histogram
        axes[0].hist(prey_mse, bins=bins, edgecolor="black", alpha=0.7)
        axes[0].set_title("Prey MSE Histogram")
        axes[0].set_xlabel("MSE pray")
        axes[0].set_ylabel("Frequency")

        # Predator Error Histogram
        axes[1].hist(predator_mse, bins=bins, edgecolor="black", alpha=0.7)
        axes[1].set_title("Predator MSE Histogram")
        axes[1].set_xlabel("MSE predator")

        plt.tight_layout()
        plt.show()


    def plot_hist_RMSE(RMSE_df, bins):
        """
        Plots separate histograms for the Mean Squared Error (RMSE) of prey and predator predictions.

        This function takes a DataFrame containing RMSE values for prey and predator predictions
        across multiple systems and visualizes their distributions using histograms.

        Parameters:
        - MSE_df (pd.DataFrame): A DataFrame containing at least two columns:
            - 'RMSE for prey': RMSE values for prey predictions.
            - 'RMSE for predator': RMSE values for predator predictions.
        - bins (int): The number of bins to use in the histograms.

        Returns:
        - None: Displays the histograms but does not return any values.

        The function creates two side-by-side histograms:
        1. The first histogram shows the distribution of MSE values for prey predictions.
        2. The second histogram shows the distribution of MSE values for predator predictions.

        Example:
        --------
        >>> import pandas as pd
        
        >>> # Example RMSE data
        >>> data = {'RMSE for prey': [0.2, 0.5, 0.7, 0.1], 'RMSE for predator': [0.3, 0.6, 0.8, 0.2]}
        >>> rmse_df = pd.DataFrame(data)

        >>> # Plot histograms with 10 bins
        >>> plot_hist_RMSE(rmse_df, bins=10)
        """
        prey_mse = RMSE_df['RMSE for prey'].tolist()
        predator_mse = RMSE_df['RMSE for predator'].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Prey Error Histogram
        axes[0].hist(prey_mse, bins=bins, edgecolor="black", alpha=0.7)
        axes[0].set_title("Prey RMSE Histogram")
        axes[0].set_xlabel("RMSE pray")
        axes[0].set_ylabel("Frequency")

        # Predator Error Histogram
        axes[1].hist(predator_mse, bins=bins, edgecolor="black", alpha=0.7)
        axes[1].set_title("Predator RMSE Histogram")
        axes[1].set_xlabel("RMSE predator")

        plt.tight_layout()
        plt.show()



def pred_vs_true_visualisation(decoded_prediction, true_values, index):

    """
    Plots predicted vs. true prey-predator population dynamics over time for a given system.

    This function compares a model's predicted time series trajectory to the true trajectory 
    for a specific system index, and visualizes the last 30 timesteps. The prey and predator 
    values are plotted over a custom time range to illustrate alignment between predictions 
    and actual dynamics.

    Parameters:
    -----------
    decoded_prediction : list or np.ndarray
        A list or array of predicted trajectories. Each element corresponds to one system
        and should be a 2D array of shape (T, 2), where T is the number of timesteps,
        and the two columns represent [prey, predator].

    true_values : list or np.ndarray
        A list or array of ground-truth trajectories with the same structure as `decoded_prediction`.

    index : int
        The index of the system to visualize.

    Returns:
    --------
    None
        Displays a matplotlib plot comparing the true and predicted prey/predator populations
        over the last 30 time steps.
    
    """

    dec_00 = decoded_prediction[index]
    traj_00 = true_values[index] 
    print('Decoded prediction shape:',dec_00.shape) 
    print('True dataset shape:',traj_00.shape)

    traj_0 = traj_00[-30:]
    dec_0 = dec_00[-30:]
    min_length = min(len(traj_0), len(dec_0))


    time_step = np.linspace(140, 200, min_length)  # Adjust time range
    prey_values = traj_0[:min_length, 0]  # Trim original prey
    predator_values = traj_0[:min_length, 1]  # Trim original predator

    predict_prey = dec_0[:min_length, 0]  # Trim predicted prey
    predict_predator = dec_0[:min_length, 1]  # Trim predicted predator

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(time_step, prey_values, label="Original Prey", color="blue", linestyle="dashed")
    plt.plot(time_step, predator_values, label="Original Predator", color="red", linestyle="dashed")

    plt.plot(time_step, predict_prey, label="Predicted Prey", color="blue")
    plt.plot(time_step, predict_predator, label="Predicted Predator", color="red")

    # Labels and Title
    plt.xlabel("Time Steps")
    plt.ylabel("Population")
    plt.title(f"Prey-Predator Dynamics Over Time, System_ID: {index+900}")
    plt.legend()

    # Show the plot
    plt.show()

def grad_norm_loss_plot(loss_list, grad_norm_list):
    """
    Plots training loss and gradient norm on a shared y-axis over training steps.

    This function visualizes both the loss and the gradient norm across training steps
    using a single y-axis and a unified legend for clarity. This style is ideal for 
    reports or presentations where simplicity and visual harmony are preferred.

    Parameters:
    -----------
    loss_list : list of float
        List of training loss values collected at each step.

    grad_norm_list : list of float
        List of gradient L2 norms collected at each training step.

    Returns:
    --------
    None
        Displays a matplotlib line plot with shared y-axis and legend.

    Notes:
    ------
    - Both lists should be the same length.
    - Useful for observing learning behavior and training stability over time.

    Example:
    --------
    >>> grad_norm_loss_plot(loss_list, grad_norm_list)
    """


    steps = list(range(len(loss_list)))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, loss_list, label='Loss', color='blue')
    plt.plot(steps, grad_norm_list, label='Gradient Norm', color='red')
    
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.title('Training Loss and Gradient Norm Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()