'''
Importing required packages
'''

import numpy as np
import h5py

class Preprocessor:
    """ 
    
    A class designed to convert time series data into strings (i.e. words), so that we can tokenise our time series data using Qwen2.5. 

    Qwen2.5 is a series of advance LLMs designed for tasks like language understanding, reasoning, coding and math. 
    It features enhanced training on 18 trillion tokens, improving fine-tuning, and supports multiple model sizes (0.5B-72B). 
    Furthermore, Qwen2.5 delivers state-of-the-art performance (AKA highest level of performance achieved in a specific field)


    """

    def scaling_operator(data, quantile_val, upper_limit): 

        """
        Scales the input data based on a quantile-derived scaling factor.

        This function computes a scaling factor using the specified quantile of the data
        and an upper limit. The data is then scaled by dividing it by this factor.

        Parameters:
        -----------
        data : array-like
            The input numerical data to be scaled.
        quantile_val : float
            The quantile value (between 0 and 1) used to determine the scaling factor.
        upper_limit : float
            The target upper limit to which the quantile value is scaled.

        Returns:
        --------
        scaled_data : array-like
            The input data scaled according to the computed scaling factor.
        scaling_factor : float
            The factor used to scale the data.

        Example:
        --------
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> scaled_data, factor = scaling_operator(data, 0.9, 10)
        >>> print(scaled_data, factor)
        """

        scaling_factor = np.quantile(data, quantile_val)/upper_limit
        scaled_data  = np.round(data/scaling_factor, 3) # After scaling round to 3 decimal places

        return scaled_data, scaling_factor


    def array_to_string(data):
        """
        Converts a DataFrame containing prey and predator values into formatted strings.

        This function groups the DataFrame by 'system_id' and converts each group's 
        numerical data (prey and predator) into a single formatted string. 
        Each pair is separated by a comma (","), and timesteps are separated by a semicolon (";").

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame with at least three columns: 'system_id', 'prey', and 'predator'.

        Returns:
        --------
        pd.Series
            A Pandas Series where each index corresponds to a 'system_id' and the values 
            are strings formatted as "prey, predator; prey, predator; ...".

        Example:
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "system_id": [0, 0, 1, 1],
        ...     "prey": [3.7, 2.9, 4.5, 3.2],
        ...     "predator": [4.1, 3.0, 2.8, 2.1]
        ... })
        >>> formatted_strings = array_to_string(df)
        >>> print(formatted_strings[0])
        '3.7,4.1;2.9,3.0'
        >>> print(formatted_strings[1])
        '4.5,2.8;3.2,2.1'
        """
        
        formatted_strings = data.groupby("system_id").apply(
            lambda group: ";".join([",".join(map(str, row)) for row in group[["prey", "predator"]].values])
        )
        
        return formatted_strings

    

    def string_to_array(formatted_string):
        """
        Converts a formatted string of prey/predator values back into a NumPy array.
        Ensures that only complete (prey, predator) pairs are included while
        ignoring invalid leading/trailing semicolons or incomplete entries.

        Parameters:
        -----------
        formatted_string : str
            A string with values formatted as "prey, predator; prey, predator; ...".

        Returns:
        --------
        np.ndarray
            A NumPy array of shape (N, 2), where each row represents a timestep 
            with prey and predator values.
        """

        # Remove any leading or trailing semicolons
        formatted_string = formatted_string.strip(";")

        # Split the string by semicolons to get each "prey, predator" pair
        pairs = formatted_string.split(";")

        # Ensure no empty strings and filter out invalid entries
        cleaned_pairs = []
        for pair in pairs:
            values = pair.split(",")  # Split each entry into numbers
            values = [v.strip() for v in values if v.strip()]  # Remove extra spaces & filter empty

            if len(values) == 2:  # Only keep valid pairs
                try:
                    cleaned_pairs.append(list(map(float, values)))  # Convert to float
                except ValueError as e:
                    print(f"Skipping invalid entry: {pair} → {e}")  # Debugging info

        # Convert the cleaned list into a NumPy array
        return np.array(cleaned_pairs)


def load_and_preprocess(data):

    """
    Loads, scales, and formats Lotka-Volterra time series data for language model training.

    This function reads population trajectories from an HDF5 file, applies quantile-based scaling,
    splits the dataset into training and validation sets, and converts each system's trajectory 
    into a semicolon-separated string format suitable for tokenization by language models.

    Parameters:
    -----------
    data : str
        Path to the HDF5 file containing the Lotka-Volterra simulation data. The file must include:
        - 'trajectories': shape (N, T, 2), where N = number of systems, T = time steps,
                          and the last dimension corresponds to (prey, predator).
        - 'time': shape (T,), the time points corresponding to the trajectories.

    Returns:
    --------
    train_texts : np.ndarray of str
        Array of stringified time series for the training set (900 systems).
        Each string represents prey and predator values per timestep, separated by semicolons,
        e.g., "3.200,2.100;4.000,2.800;...".

    val_texts : np.ndarray of str
        Array of full-length validation sequences (100 systems), formatted as strings.

    val_texts_70 : np.ndarray of str
        Array of truncated validation sequences containing only the first 70 timesteps per system,
        useful for forecasting or partial input evaluation scenarios.

    Notes:
    ------
    - The data is scaled using the 90th percentile value (divided by 10) to normalize magnitudes.
    - All floating-point values are rounded to 3 decimal places for string formatting.
    - This format is intended for use with LLM-style tokenization pipelines (e.g., Qwen2.5).
    """
    
    #Load data
    with h5py.File(data, 'r') as f:
        # Access the full dataset
        trajectories = f['trajectories'][:]
        time_points = f['time'][:]
    
    #Scaling data
    scaling_factor = np.quantile(trajectories, 0.9)/10
    trajectories_scaled  = np.round(trajectories/scaling_factor, 3) # After scaling round to 3 decimal places

    # Split data into Training and Validation sets

    traj_train = trajectories_scaled[:900, :, :]
    traj_val = trajectories_scaled[-100:, :, :]
    traj_val_70 = traj_val[:, :70, :]

    # Convert arrays to string

    train_texts = np.array([
        ";".join([f"{prey:.3f},{pred:.3f}" for prey, pred in system])
        for system in traj_train
    ])
    

    val_texts = np.array([
        ";".join([f"{prey:.3f},{pred:.3f}" for prey, pred in system])
        for system in traj_val
    ])

    val_texts_70 = np.array([
        ";".join([f"{prey:.3f},{pred:.3f}" for prey, pred in system])
        for system in traj_val_70
    ])

    return train_texts, val_texts, val_texts_70



def data_scale_split(data):
    """
    Loads, scales, and splits Lotka-Volterra time series data into training and validation sets.

    This function reads simulation data from an HDF5 file, applies quantile-based scaling to normalize
    population values, and returns split datasets for training and validation. It also includes a 
    truncated version of the validation set with fewer time steps for use in forecasting tasks.

    Parameters:
    -----------
    data : str
        Path to the HDF5 file containing the Lotka-Volterra data. The file must contain:
        - 'trajectories': A NumPy array of shape (N, T, 2), where N is the number of systems,
          T is the number of time steps, and the last dimension represents (prey, predator) values.
        - 'time': A 1D array of time points (not used directly but typically aligned with trajectories).

    Returns:
    --------
    traj_train : np.ndarray
        Scaled training trajectories for 900 systems, shape (900, T, 2).

    traj_val : np.ndarray
        Full-length validation trajectories for 100 systems, shape (100, T, 2).

    traj_val_70 : np.ndarray
        Truncated validation trajectories with only the first 70 time steps per system,
        shape (100, 70, 2), useful for partial input prediction.

    Notes:
    ------
    - Data is scaled using the 90th percentile of all values divided by 10.
    - Scaling helps normalize across varying system magnitudes.
    - All values are rounded to 3 decimal places after scaling.
    """

        #Load data
    with h5py.File(data, 'r') as f:
        # Access the full dataset
        trajectories = f['trajectories'][:]
        time_points = f['time'][:]
    
    #Scaling data
    scaling_factor = np.quantile(trajectories, 0.9)/10
    trajectories_scaled  = np.round(trajectories/scaling_factor, 3) # After scaling round to 3 decimal places

    # Split data into Training and Validation sets

    traj_train = trajectories_scaled[:900, :, :]
    traj_val = trajectories_scaled[-100:, :, :]
    traj_val_70 = traj_val[:, :70, :]

    return traj_train, traj_val, traj_val_70


def sequence_length_array(tokenized_string):
    """
    Computes the sequence lengths for a list of tokenized inputs.

    This function takes a list of tokenized input dictionaries (as returned by a Hugging Face tokenizer 
    with `return_tensors="pt"`), and extracts the length (number of tokens) for each input sequence.

    Parameters:
    -----------
    tokenized_string : list of dict
        A list where each item is a dictionary containing tokenized data with at least the key 
        'input_ids', whose value is a tensor of shape (1, sequence_length).

    Returns:
    --------
    max_lengths : np.ndarray
        A 1D NumPy array containing the sequence lengths (number of tokens) for each input.

    Example:
    --------
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    >>> encoded = [tokenizer(text, return_tensors="pt") for text in text_list]
    >>> lengths = sequence_length_array(encoded)
    """
    max_lengths = np.array([entry["input_ids"].shape[1] for entry in tokenized_string])
    return max_lengths


