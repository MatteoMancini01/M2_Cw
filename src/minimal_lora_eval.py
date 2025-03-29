# minimal_lora_eval.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from .qwen import*



# LoRA module
class LoRALinear(nn.Module):

    """
    A module that adds trainable low-rank adapters (LoRA) to an existing linear layer.

    LoRA works by injecting a low-rank update (B @ A) to the frozen original linear transformation.
    This allows fine-tuning of large models by only training a small number of parameters.

    Parameters:
    -----------
    original_linear : nn.Linear
        The original linear layer to wrap with LoRA.
    r : int
        The rank of the low-rank matrices A and B (i.e., the LoRA bottleneck).
    alpha : int, optional
        A scaling factor for the LoRA output. Defaults to r (i.e., no scaling).

    Attributes:
    -----------
    A : nn.Parameter
        Trainable low-rank matrix of shape (r, in_features), initialized with He init.
    B : nn.Parameter
        Trainable low-rank matrix of shape (out_features, r), initialized as zeros.

    Forward:
    --------
    Applies the frozen base linear transformation and adds the scaled LoRA output.
    """

    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r
        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")
    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)

# Inject LoRA
model, tokenizer = load_qwen()
lora_rank = 4
for layer in model.model.layers:
    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

def process_sequences(texts, tokenizer, max_length=512, stride=256):

    """
    Tokenizes and chunks a list of text sequences into overlapping windows suitable
    for language model training.

    Parameters:
    -----------
    texts : list of str
        Time series represented as text strings.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer compatible with the target language model (e.g., Qwen2.5).
    max_length : int
        Maximum sequence length for model input.
    stride : int
        Step size to slide the window for overlapping chunks.

    Returns:
    --------
    Tensor
        A tensor of shape (N, max_length) where N is the number of generated chunks.
    """

    all_input_ids = []
    for text in texts:
        # Apply Qwen's tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                    ]
                )
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)




def evaluate_loss(model, val_loader):
    """
    Computes the average validation loss (cross-entropy) over the validation dataset.

    This function evaluates the model's language modeling performance by computing 
    the token-level cross-entropy loss on the provided validation set. It uses the 
    same loss formulation as during training and runs in no-grad mode to avoid 
    gradient computation. A progress bar is displayed using tqdm to monitor progress.

    Parameters:
    -----------
    model : torch.nn.Module
        The language model (e.g., Qwen with LoRA adapters) to evaluate.

    val_loader : torch.utils.data.DataLoader
        DataLoader providing batches of tokenized validation input sequences.

    Returns:
    --------
    float
        The average validation loss computed over all batches.

    Example:
    --------
    >>> val_loss = evaluate_loss(model, val_loader)
    >>> print(f"Validation Loss: {val_loss:.4f}")
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(val_loader, desc="Validating")

    with torch.no_grad():
        for (batch,) in progress_bar:
            outputs = model(batch, labels=batch)
            loss = outputs.loss.item()

            total_loss += loss
            total_batches += 1

            avg_loss = total_loss / total_batches
            progress_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return total_loss / total_batches
