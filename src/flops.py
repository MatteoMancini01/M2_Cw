
def total_transformer_training_flops(num_steps, batch_size, seq_len, num_layers, d_model, num_heads, intermediate_dim, lora_rank=0):
    """
    Compute total FLOPs for training a Transformer with LoRA.

    Parameters:
    -----------
    num_steps : int
        Total number of training steps.
    batch_size : int
        Number of sequences per batch.
    seq_len : int
        Sequence length.
    num_layers : int
        Number of Transformer layers.
    d_model : int
        Hidden size.
    num_heads : int
        Number of attention heads.
    intermediate_dim : int
        Dimension of the feedforward layer (usually 2* or 4* d_model for SwiGLU/FFN).
    lora_rank : int
        LoRA rank (0 to disable LoRA FLOPs).

    Returns:
    --------
    total_flops : int
        Estimated total FLOPs over the training run.
    """

    d_k = d_model // num_heads
    total_flops = 0

    # Self-Attention FLOPs per layer
    attn_proj = 3 * d_model * d_model * seq_len  # Q, K, V projections
    qk_matmul = seq_len * d_k * seq_len          # QK^T
    attn_weighted_v = seq_len * d_k * seq_len    # Attn × V
    out_proj = d_model * d_model * seq_len       # Output projection
    attention_flops = attn_proj + qk_matmul + attn_weighted_v + out_proj

    # LoRA (if used): two low-rank projections per injected layer (Q, V)
    lora_flops = 0
    if lora_rank > 0:
        lora_flops_q = 2 * lora_rank * d_model * seq_len
        lora_flops_v = 2 * lora_rank * d_model * seq_len
        lora_flops = lora_flops_q + lora_flops_v

    # Feedforward FLOPs per layer (SwiGLU style)
    ffn_proj1 = d_model * intermediate_dim * seq_len  # Linear for gate
    ffn_proj2 = d_model * intermediate_dim * seq_len  # Linear for activation
    ffn_act = intermediate_dim * seq_len              # Swish activation
    ffn_mult = intermediate_dim * seq_len             # Gating mult
    ffn_proj_out = intermediate_dim * d_model * seq_len
    ffn_flops = ffn_proj1 + ffn_proj2 + ffn_act + ffn_mult + ffn_proj_out

    # LayerNorm FLOPs (2 × d_model × seq_len per layer)
    norm_flops = 2 * d_model * seq_len * 2

    # Total FLOPs per sequence per layer
    flops_per_layer = attention_flops + ffn_flops + norm_flops + lora_flops

    # Multiply by number of layers and batch size
    flops_per_step = flops_per_layer * num_layers * batch_size

    # Account for forward + backward
    total_flops = num_steps * flops_per_step * 3

    return total_flops

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
    # Extract model parameters
    num_heads = 14 # Number of attention heads 
    d_model = 896 # d_model hidden size
    num_layers = 24 # Number of Transformer layers

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