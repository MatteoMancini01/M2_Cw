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
        scaled_data  = data/scaling_factor

        return scaled_data, scaling_factor

