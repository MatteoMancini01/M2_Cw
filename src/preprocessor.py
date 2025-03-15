'''
Importing required packages
'''

import numpy as np


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

        Parameters:
        -----------
        formatted_string : str
            A string with values formatted as "prey, predator; prey, predator; ...".

        Returns:
        --------
        np.ndarray
            A NumPy array of shape (100, 2), where each row represents a timestep 
            with prey and predator values.

        Example:
        --------
        >>> formatted_str = "3.757031, 4.115786;2.928965, 3.083176;2.698359, 2.232228"
        >>> array = string_to_array(formatted_str)
        >>> print(array.shape)  # Output: (100, 2)
        """
        
        # Split the string by semicolons to get each "prey, predator" pair
        pairs = formatted_string.split(";")
        
        # Convert each pair into a tuple of float values
        array = np.array([list(map(float, pair.split(","))) for pair in pairs])
        
        return array
            

    
