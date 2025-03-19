
# Extract model parameters
num_heads = 14 # Number of attention heads 
d_model = 896 # d_model hidden size
num_layers = 24 # Number of Transformer layers

def flops_mlp(num_layers, num_hidden, input_dim):
    """Compute the FLOPs for a given MLP"""
    flops = 0
    layer_dims = [input_dim] + [num_hidden] * num_layers

    for i in range(num_layers):
        # matrix multiplication:
        flops += (2 * layer_dims[i] - 1) * layer_dims[i + 1]
        if i < num_layers - 1:
            # ReLU activation (except last layer)
            flops += layer_dims[i + 1]

    return flops

def flops_forward_pass(batch_size, num_layers, num_hidden, input_dim):
    """Compute the FLOPs for a forward pass in the training loop."""
    return batch_size * flops_mlp(num_layers, num_hidden, input_dim)

def flops_forward_and_back(batch_size, num_layers, num_hidden, input_dim):
    # (We simplify things and assume backward = 2x forward, as per instruction in coursework)
    return 3 * flops_forward_pass(batch_size, num_layers, num_hidden, input_dim)

def total_flops(num_steps_training, batch_size, num_layers, num_hidden, input_dim):
    """Total FLOPs for the experiment"""
    return num_steps_training * flops_forward_and_back(batch_size, num_layers, num_hidden, input_dim)

# For part 2b
def compute_generation_flops(input_ids_list, max_lengths):

    """
    Computes the total number of floating point operations (FLOPs) required for generating text
    using a Transformer-based model like Qwen2.5.

    This function estimates FLOPs for:
    1. Self-Attention mechanism (including QK^T computation and output projection).
    2. Feedforward layers (SwiGLU activation).
    3. Layer Normalization.
    
    The FLOPs are calculated per token and then multiplied by the total number of generated tokens.

    Parameters:
    -----------
    input_ids_list : list of list[int]
        A list of input sequences, where each sequence is a list of token IDs.
    
    max_lengths : list[int]
        A list of maximum token lengths for each sequence, used to determine the total number of generated tokens.

    Returns:
    --------
    total_flops : int
        The total number of FLOPs required for generating all sequences in `input_ids_list`.

    Notes:
    ------
    - The function assumes that each sequence undergoes `2 * max_length` token generations.
    - This computation is specific to a Transformer model with `num_layers` layers, `num_heads` attention heads, 
      and `d_model` hidden dimension.
    - FLOPs from sampling strategies (e.g., top-k, nucleus sampling) are considered negligible and not included.

    Example:
    --------
    >>> input_ids_list = [[101, 2001, 2023, 102], [101, 2033, 2054, 2003, 1996, 2087, 102]]
    >>> max_lengths = [10, 15]
    >>> compute_generation_flops(input_ids_list, max_lengths)
    3.5e+12  # Example output (FLOPs count)
    """

    total_flops = 0
    
    for i in range(len(input_ids_list)):
        seq_length = len(input_ids_list[i])  # Input sequence length
        generated_tokens = 2*max_lengths[i]  # Total tokens generated
        
        # Self-Attention FLOPs
        attention_flops = num_heads*seq_length*d_model*(2*d_model - 1)
        attention_flops += seq_length**2*d_model  # QK^T computation
        attention_flops += seq_length*d_model*(2*d_model - 1)  # Output projection

        # Feedforward FLOPs (SwiGLU)
        feedforward_flops = 2*seq_length*d_model*(2*d_model - 1)
        
        # LayerNorm FLOPs
        norm_flops = 2 * seq_length * d_model

        # Total FLOPs per token
        flops_per_token = num_layers * (attention_flops + feedforward_flops + norm_flops)
        
        # Total FLOPs for the entire sequence
        flops_for_sequence = generated_tokens * flops_per_token

        # Accumulate total FLOPs
        total_flops += flops_for_sequence

    return total_flops