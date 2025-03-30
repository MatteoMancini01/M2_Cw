import torch
from tqdm import tqdm
import torch.nn as nn
from .qwen import tokenize_time_series_np
from sklearn.metrics import mean_absolute_error
import pandas as pd

# TensorDataset allows you to wrap tensors into a dataset object
# DataLoader provides minibatch iteration over the dataset
from torch.utils.data import TensorDataset, DataLoader

# Accelerator from Hugging Face simplifies GPU/multi-device/mixed-precision training
from accelerate import Accelerator

from .preprocessor import*
from .plotting import*

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
        assert isinstance(original_linear, nn.Linear)
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
        
        # Initialise A with He initialization
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)
    

def train_lora_model(model, tokenizer, lora_rank=4, learning_rate=1e-5, batch_size=4, max_ctx_length=512,train_steps=5000):
    
    """
    Fine-tunes a Qwen2.5 language model on Lotka-Volterra time series data using LoRA and returns final training loss.

    This function applies Low-Rank Adaptation (LoRA) to the self-attention query and 
    value projections of a Qwen2.5 transformer model. It then trains the model using 
    next-token prediction on tokenized time series data derived from time-series trajectories.
    The training process is optimized using HuggingFace's `Accelerator` for efficient device handling.

    Parameters:
    -----------
    model : transformers.PreTrainedModel
        A pre-initialized Qwen2.5 model (e.g., from `load_qwen()`), which will be 
        modified in-place by applying LoRA to its attention layers.

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer corresponding to the Qwen model. Used for encoding time-series strings 
        into token ID sequences suitable for training.

    lora_rank : int, default=4
        The rank of the LoRA low-rank adaptation matrices. Higher ranks increase model 
        capacity and computational cost.

    learning_rate : float, default=1e-5
        The learning rate for the Adam optimizer used during training.

    batch_size : int, default=4
        The number of tokenized sequences processed per training batch.

    max_ctx_length : int, default=512
        The maximum sequence length (in tokens) for each input segment passed to the model.

    train_steps : int, default=5000
        The total number of training steps to execute. Training will stop once this 
        number of steps is reached, regardless of dataset size.

    Returns:
    --------
    model : transformers.PreTrainedModel
        The trained model with LoRA adapters applied. Returned in evaluation mode 
        (`model.eval()` has been called) and ready for inference or further evaluation.

    final_loss : float
        The final training loss (cross-entropy) from the last processed batch. Useful 
        for comparing model performance across hyperparameter configurations.

    Notes:
    ------
    - The model is modified in-place; avoid passing in a previously LoRA-wrapped model unless guarded.
    - Only the final batch loss is returned. If average or validation loss is desired, it should be 
      computed separately.
    - Assumes that the time-series data is available at 'data/lotka_volterra_data.h5'.
    """

    # Applay LoRA to model
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    # Load and process data
    train_texts, val_texts, _ = load_and_preprocess("data/lotka_volterra_data.h5")
    train_input_ids = process_sequences(train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2)
    val_input_ids = process_sequences(val_texts, tokenizer, max_ctx_length, stride=max_ctx_length)

    # Prepare data loaders
    train_dataset = TensorDataset(train_input_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    # Accelerator setup
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

        # Training loop
    model.train()
    steps = 0
    final_loss = None

    while steps < train_steps:  # QPLPPP = 10,000 steps
        for (batch,) in tqdm(train_loader, desc=f"Steps {steps}"):

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass with labels = input_ids for language modeling (next-token prediction)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            final_loss = loss.item()

            # Backpropagation using Accelerator
            accelerator.backward(loss)
            optimizer.step()

            # Step counter
            steps += 1

            # Break loop after desired number of steps
            if steps >= train_steps:
                break
    model.eval()

    return model, final_loss



def evaluate_loss_perplexity_val(trained_model, tokenizer, val_texts, batch_size,  max_length=512):
    """
    Computes average validation loss and perplexity for a fine-tuned language model.

    This function processes a list of validation sequences using a tokenizer, chunks 
    them into fixed-length windows suitable for the model, and computes:
        - The average cross-entropy loss (per-token)
        - The corresponding perplexity (exp(loss))

    It uses the same loss formulation as during training (i.e., next-token prediction) 
    and runs in no-grad mode to avoid gradient computation. A `tqdm` progress bar 
    is shown to track evaluation.

    Parameters:
    -----------
    trained_model : transformers.PreTrainedModel
        The trained language model in evaluation mode (e.g., Qwen2.5 with LoRA applied).

    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer corresponding to the trained model, used to process val_texts.

    val_texts : list or np.ndarray of str
        A list or array of time-series strings representing the validation set. 
        Each entry should be a string of semicolon-separated "prey,predator" pairs.

    batch_size : int
        Number of sequences per batch during evaluation.

    Returns:
    --------
    loss_val : float
        The average token-level cross-entropy loss on the validation set.

    perplexity_val : float
        The perplexity computed as `exp(loss_val)`, a standard metric in language modeling.

    Example:
    --------
    >>> loss, ppl = evaluate_loss_perplexity_val(model, tokenizer, val_texts, batch_size=4)
    >>> print(f"Validation Loss: {loss:.4f}, Perplexity: {ppl:.2f}")"
    """
    

    val_input_ids = process_sequences(val_texts, tokenizer, max_length=max_length, stride=max_length)

    accelerator = Accelerator()

    trained_model.eval()
    total_loss = 0.0
    total_batches = 0

    val_dataset = TensorDataset(val_input_ids)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    val_loader = accelerator.prepare(val_loader)

    progress_bar = tqdm(val_loader, desc="Validating")

    with torch.no_grad():
        for (batch,) in progress_bar:
            outputs = trained_model(batch, labels=batch)
            loss = outputs.loss.item()

            total_loss += loss
            total_batches += 1

            avg_loss = total_loss / total_batches
            progress_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
    
    loss_val = total_loss / total_batches
    perplexity_val = np.exp(loss_val)

    return loss_val, perplexity_val


def prediction_after_training(trained_model, tokenizer, val_texts_70):

    """
    Generates autoregressive forecasts from a fine-tuned Qwen2.5 model using truncated validation inputs.

    This function takes a trained language model and a set of validation time series inputs 
    (first 70 timesteps), tokenizes them using the provided tokenizer, and uses 
    `model.generate()` to forecast the future trajectory in token space.

    Parameters:
    -----------
    trained_model : transformers.PreTrainedModel
        A fine-tuned Qwen2.5 language model, potentially wrapped with `Accelerator`. 
        The model will be unwrapped and moved to GPU automatically for inference.

    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer corresponding to the Qwen2.5 model. Used to encode and decode time series strings.

    val_texts_70 : list or np.ndarray of str
        A collection of validation time series strings, each containing only the first 70 timesteps.
        Each string is formatted as `"prey,predator;prey,predator;..."`.

    Returns:
    --------
    predictions_encoded : list of torch.Tensor
        A list of token sequences generated by the model. Each tensor contains the full output 
        (original input + generated continuation). You can decode them later using:
        `tokenizer.decode(output[0], skip_special_tokens=True)`.

    Notes:
    ------
    - The generation uses greedy decoding (`do_sample=False`) for reproducibility.
    - `min_new_tokens=350` and `max_new_tokens=400` are set to control output length.
    - The model is put in `eval` mode and moved to CUDA device automatically.
    - Padding and attention masking are respected via the tokenized inputs.
    - This function returns raw generated tokens; decoding and post-processing are up to the caller.

    Example:
    --------
    >>> predictions = prediction_after_training(model, tokenizer, val_texts_70)
    >>> decoded = [tokenizer.decode(seq[0], skip_special_tokens=True) for seq in predictions]
    >>> print(decoded[0])  # Output for first validation system
    """

    accelerator = Accelerator()

    tok_val_texts_70 = tokenize_time_series_np(val_texts_70)



    # Unwrap the model from Accelerate (optional) and move to GPU
    trained_model = accelerator.unwrap_model(trained_model)
    trained_model.to("cuda")
    trained_model.eval()

    predictions_encoded = []

    # Loop over each validation sample for generation
    for i in tqdm(range(len(tok_val_texts_70)), desc="Generating predictions"):
        
        # Extract tokenized input and attention mask for system i
        input_ids = tok_val_texts_70[i]["input_ids"].to(trained_model.device)
        attention_mask = tok_val_texts_70[i]["attention_mask"].to(trained_model.device)

        # Use model.generate() to predict the next tokens
        output = trained_model.generate(
            input_ids=input_ids,                 # Input sequence (first 70 timesteps)
            attention_mask=attention_mask,       # Mask to ignore padding tokens
            min_new_tokens=350,                   # Force at least 30 new tokens to be generated
            max_new_tokens=400,                   # Upper bound to avoid runaway generation
            do_sample=False,                     # Use deterministic decoding (greedy search)
            eos_token_id=tokenizer.eos_token_id  # Stop generation at EOS if it appears
        )

        # Append generated output to the list (will decode later)
        predictions_encoded.append(output)

    return predictions_encoded



def decoder_and_metrics_evaluator(predicted_encoded, tokenizer):

    """
    Decodes model-generated token sequences and evaluates forecasting accuracy 
    against ground truth Lotka-Volterra dynamics.

    This function performs several key steps:
    1. Decodes the generated token sequences into LLMTIME-formatted strings.
    2. Converts the decoded strings back into numerical arrays.
    3. Extracts timesteps 70-100 from both predicted and true validation data.
    4. Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) 
       for prey and predator values.
    5. Computes signed per-timestep prediction errors (true - predicted) per system.

    Parameters:
    -----------
    predicted_encoded : list of torch.Tensor
        A list of generated token sequences, one for each validation system. Each sequence 
        includes the model's autoregressive forecast after the 70-timestep input prompt.

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used for decoding the predicted token sequences back into text.

    Returns:
    --------
    predictions_decoded : list of np.ndarray
        Decoded full-length numerical arrays converted from the generated token sequences.

    predicted_output : list of np.ndarray
        The last 30 timesteps (t=70 to t=100) of each decoded prediction array.

    true_values : list of np.ndarray
        The ground truth validation trajectories for each system, sliced from timestep 70 onward.

    MSE_values : list of [float, float]
        Per-system MAE values: [prey MAE, predator MAE].

    RMSE_values : np.ndarray
        Per-system RMSE values: same shape as MSE_values.

    error_per_system : list of [np.ndarray, np.ndarray]
        Signed prediction errors for each system:
        - error_per_system[i][0]: prey errors (true - pred)
        - error_per_system[i][1]: predator errors (true - pred)

    Notes:
    ------
    - This function assumes that model predictions are long enough to cover at least 
      timesteps 70 to 100 after decoding.
    - Ground truth data is loaded internally from "data/lotka_volterra_data.h5".
    - Errors and metrics are robust to varying lengths and will truncate mismatched arrays.

    Example:
    --------
    >>> decoded, preds, truths, mae, rmse, errors = decoder_and_metrics_evaluator(predicted_tokens, tokenizer)
    >>> print("Prey RMSE for system 0:", rmse[0][0])
    >>> print("Signed errors (predator) for system 0:", errors[0][1])
    """

    string_to_array = Preprocessor.string_to_array

    _, true_val_values, _ = data_scale_split("data/lotka_volterra_data.h5")

    predictions_decoded = []

    for i in range(len(predicted_encoded)):
        decoded_output = string_to_array(tokenizer.decode(predicted_encoded[i][0], skip_special_tokens=True))
        predictions_decoded.append(decoded_output)


    predicted_output = []
    true_values = []
    # Collecting the last 30 pair from each array
    for i in range(len(predictions_decoded)):
        sub_output = predictions_decoded[i]
        sub_output_min_30 = sub_output[70:100]
        true_v = true_val_values[i]
        true_v_min_30 = true_v[70:]
        predicted_output.append(sub_output_min_30)
        true_values.append(true_v_min_30)

        # Computing MSE
    MSE_values = []

    for i in range(len(true_values)):

        # Truncate to match the shorter list
        min_length = min(len(predicted_output[i]), len(true_values[i]))
        pr_out = predicted_output[i][:min_length]
        true_val = true_values[i][:min_length]

        # Compute the MSE
        mse_prey = mean_absolute_error(pr_out[:,0], true_val[:,0]) # Computing MSE for prey
        mse_predator = mean_absolute_error(pr_out[:,1], true_val[:,1]) # Computing MSE for predator
        MSE_values.append([mse_prey, mse_predator])

    # Computing RMSE
    RMSE_values = np.sqrt(MSE_values)

    # Computing the error for each system

    error_per_system = []

    for i in range(len(true_values)):

        # Truncate to match the shorter list
        min_length = min(len(predicted_output[i]), len(true_values[i]))
        pr_out = predicted_output[i][:min_length]
        true_val = true_values[i][:min_length]

        # Computing errors
        prey_error = true_val[:,0] - pr_out[:,0]
        predator_error = true_val[:,1] - pr_out[:,1]
        
        # Collecting errors into a list
        error_per_system.append([prey_error, predator_error])

    return predictions_decoded, predicted_output, true_values, MSE_values, RMSE_values, error_per_system


def collective_plots(predicted_encoded, tokenizer, system_id=0, bins=30):
    """
    Generates a suite of diagnostic plots to visualise model performance on Lotka-Volterra forecasts.

    This function decodes model-generated sequences, computes errors and metrics, and produces 
    the following visualisations:
    
    1. Histogram of Mean Absolute Error (MSE) for prey and predator across all systems.
    2. Histogram of Root Mean Squared Error (RMSE) for prey and predator across all systems.
    3. Signed error histograms (true - predicted) for the specified `system_id`.
    4. Predicted vs. true time series plot for the specified `system_id` (timesteps 70-100).

    Parameters:
    -----------
    system_id : int, default=0
        The ID of the system to use for individual error visualisations and trajectory comparison.

    bins : int, default=30
        Number of bins to use in the error histograms.

    predicted_encoded : list of torch.Tensor
        The list of generated token sequences for all validation systems, as returned 
        from `model.generate()` during inference.

    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used to decode generated outputs from token IDs to text format.

    Returns:
    --------
    None
        Displays plots directly using matplotlib. No values are returned.

    Notes:
    ------
    - This function internally calls `decoder_and_metrics_evaluator` to decode predictions 
      and compute error metrics.
    - Metrics (MSE, RMSE) and errors are aggregated into `DataFrame`s for visualisation.
    - Assumes validation ground truth is accessible from the default data path.
    - All plots are drawn in sequence and may block script execution in notebook environments.

    Example:
    --------
    >>> collective_plots(system_id=3, bins=40, predicted_encoded=predictions, tokenizer=tokenizer)
    """

    predictions_decoded, _, true_values, MSE_values, RMSE_values, error_per_system = decoder_and_metrics_evaluator(predicted_encoded, tokenizer)

    df_MSE_values = pd.DataFrame({
        "system_id": np.arange(len(MSE_values)),  
        "MSE for prey": np.array(MSE_values)[:, 0],  # Flatten prey values
        "MSE for predator": np.array(MSE_values)[:, 1]  # Flatten predator values
    })
    df_RMSE_values = pd.DataFrame({
        "system_id": np.arange(len(MSE_values)),  
        "RMSE for prey": RMSE_values[:, 0],  # Flatten prey values
        "RMSE for predator": RMSE_values[:, 1]  # Flatten predator values
    })

    PlotProject.plot_hist_MSE(df_MSE_values,bins=bins) # Plotting MSE distribution
    PlotProject.plot_hist_RMSE(df_RMSE_values,bins=bins) # Plotting RMSE distribution
    PlotProject.plot_error_hist_system(error_per_system, system_id, bins=bins)
    pred_vs_true_visualisation(predictions_decoded, true_values, system_id)