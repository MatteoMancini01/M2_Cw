import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen():

    """
    Loads the Qwen2.5-0.5B-Instruct language model and tokenizer from Hugging Face.

    This function initialises the Qwen2.5 model and tokenizer, freezes all model parameters
    except for the language model (LM) head bias, making it the only trainable parameter.

    Returns:
    --------
    model : AutoModelForCausalLM
        The pre-trained Qwen2.5-0.5B-Instruct model with a trainable LM head bias.
    tokenizer : AutoTokenizer
        The corresponding tokenizer for processing text input.

    Example:
    --------
    >>> model, tokenizer = load_qwen()
    >>> text = "Hello, how are you?"
    >>> input_ids = tokenizer(text, return_tensors="pt").input_ids
    >>> output = model.generate(input_ids, max_length=50)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    """


    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)
    )
    model.lm_head.bias.requires_grad = True

    return model, tokenizer


model, tokenizer = load_qwen()


def tokenize_time_series(formatted_strings):
    """
    Tokenises a Series of formatted time series strings using the Qwen tokeniser.

    Parameters:
    -----------
    formatted_strings : pd.Series
        A Pandas Series where each value is a formatted time series string.

    Returns:
    --------
    list of torch.Tensor
        A list of tokenised tensors ready for model input.
    
    Example:
    --------
    >>> tokenized_data = tokenize_time_series(formatted_strings)
    >>> print(tokenized_data[0])  # Example of tokenized output
    """
    
    tokenized_data = formatted_strings.apply(
        lambda x: tokenizer(x, return_tensors="pt", padding=True, truncation=True)
    )
    
    return tokenized_data

def tokenize_time_series_np(formatted_strings):
    """
    Tokenizes a NumPy array of formatted time series strings using a Hugging Face-compatible tokenizer.

    Parameters:
    -----------
    formatted_strings : np.ndarray or list of str
        A 1D NumPy array or list where each element is a formatted time series string 
        (e.g., "0.300,0.500;0.275,0.490;...").

    Returns:
    --------
    list of dict
        A list of dictionaries, each containing tokenized tensors 
        (e.g., 'input_ids', 'attention_mask') suitable for model input using PyTorch.

    Example:
    --------
    >>> tokenized_data = tokenize_time_series_np(formatted_strings)
    >>> print(tokenized_data[0]['input_ids'].shape)  # Shape of tokenized input tensor
    """
    
    tokenized_data = [
        tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        for x in formatted_strings
        ]
    
    return tokenized_data