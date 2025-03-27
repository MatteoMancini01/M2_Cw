<b>zero-shot</b>
Zero-shot refers to a scenario is when a model is trained to categorise or regosnise objects or concepts without having seen any of those categories or objects, purely relying on pre-exisiting knowledge or semantic relationships.

<b>Padding</b>
This techinque is used in CNNs, to preserve the spatial dimensions of the input image after convolution operations. For an image, padding consists in adding extra pixels around the border of the input feature map before applying the convolution operation.

<b>Attention mask</b>
When using tokenised input, some sequences are shorter than others, so we pad them to the same length. The attention mask tells the model which parts of the sequence are real input vs padding.

<b>`input_ids`</b>
The `input_ids` are generated when you tokeinse text using transformer model's tokeniser.


### Repo structure

1. Introduction:
    Expalein what the objective of the project is, plan of action, and execution

2. Background:
    Quick explenation of LLMTIME, Qwen2.5 and LoRA

3. Baseline

4. LoRA

5. Conclusions

### How `lora_skeleton.py` works
- The original linear layer is frozen (non-trainable).
- Two new trainable matrices `A` and `B` are introduced.
- Output: `original(x) + (x @ A.T) @ B.T * (alpha/r)`


`process_sequences` function, how it works:
- Each long string is tokenised into IDs.
- Then it's split into chunks of length `max_length`, overlapping by `stride`.
- Shorter final chunks are padded.

<b>Training loop</b>
```bash
    while steps < 10000:
        for (batch,) in train_loader:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            steps += 1
```
How it works:

- Loops unitl `steps` hits 10000.
- For each batch to tokenised sequences:
    - Forward pass through the model 
    - Compute loss against itself (casual LM objective)
    - Backpropagate and update only LoRA weights
    - Use `accelerate` to handle device placement and mixed precision efficiently
