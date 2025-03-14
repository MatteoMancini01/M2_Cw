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
