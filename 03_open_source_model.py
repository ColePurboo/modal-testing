import modal                                            # Modal SDK: lets us run a Python function on a rented cloud GPU

app = modal.App("open-source-model")                    # a named group of remote functions; shows up in the Modal dashboard
image = modal.Image.debian_slim().pip_install(          # describe the software on the remote machine: minimal Linux plus...
    "torch", "transformers", "accelerate"               # ...PyTorch (tensors + GPU), transformers (model/tokenizer loaders), accelerate (device helpers)
)                                                       # Modal builds this once (~2 min), caches it, and reuses it on every later run

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"               # Hugging Face ID of the model: ~0.5 billion parameters, chat-tuned, public, no token needed


@app.function(image=image, gpu="T4", memory=8192, timeout=900)   # run this function remotely: our image, one T4 GPU, 8 GB RAM, give up after 15 min
def inspect_and_generate(prompt: str):                  # everything inside here executes on the remote GPU machine, not on the Mac
    import torch                                        # imported inside: torch only exists on the remote image, not needed locally
    from transformers import AutoTokenizer, AutoModelForCausalLM   # Auto* classes read the model's config and pick the right tokenizer/architecture

    out = {}                                            # results dict; gets sent back to the laptop at the end

    # 1. Tokenizer: text -> list of integer token IDs, and back
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)     # download the tokenizer files (vocab + merge rules) and build the text<->ID converter
    ids = tok(prompt)["input_ids"]                      # split the prompt into chunks and map each chunk to its integer ID, e.g. "Ex" -> 840
    out["token_ids"] = ids                              # the integers: the ONLY thing the model ever sees
    out["tokens"] = tok.convert_ids_to_tokens(ids)      # the same IDs turned back into their text chunks, so we can read them

    # 2. Model: download the weights and build the architecture, in 16-bit
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)   # build the layer structure from config, fill every named weight slot from the downloaded file, store as 2-byte floats in CPU RAM
    model = model.to("cuda")                            # copy all ~1 GB of weights from CPU RAM across PCIe into GPU VRAM (one time)
    model.eval()                                        # inference mode: switches off training-only behaviour (e.g. dropout)

    # 3. Look inside: the layers and the parameter count
    out["architecture"] = str(model)                    # the printable tree of layers: embedding, 24 decoder blocks of Linear layers, lm_head
    n_params = sum(p.numel() for p in model.parameters())   # count every learnable number across every weight table and bias
    out["n_params"] = n_params                          # ~494 million dials
    out["predicted_weights_gb"] = round(n_params * 2 / 1e9, 3)    # rule of thumb: parameters x bytes per parameter (2 bytes each in fp16)
    out["actual_vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)   # what PyTorch really holds in VRAM right now; should match the prediction
    out["first_weight_name_and_shape"] = [              # a weights file is a dict of name -> number table; peek at the first four entries
        (name, tuple(p.shape)) for name, p in list(model.named_parameters())[:4]   # e.g. ('model.embed_tokens.weight', (151936, 896))
    ]

    # 4. One forward pass: what does the model output?
    messages = [{"role": "user", "content": prompt}]    # chat-style input: a list of who-said-what
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)   # wrap it in the model's chat format (system/user/assistant markers) as plain text
    inputs = tok(text, return_tensors="pt").to("cuda")  # tokenize to a PyTorch tensor of shape [1, seq_len] and copy it RAM -> VRAM (must be on the same device as the model)
    with torch.no_grad():                               # no training here, so don't record the autograd receipt (saves memory and time)
        logits = model(**inputs).logits                 # forward pass: one score per vocabulary token at every position -> shape [batch, seq_len, vocab_size]
    out["logits_shape"] = tuple(logits.shape)           # e.g. (1, 39, 151936): 1 example, 39 input tokens, 151936 possible next tokens

    # 5. Generate one token at a time, by hand
    cur = inputs["input_ids"]                           # the growing sequence of token IDs; starts as just the prompt
    generated = []                                      # the new token IDs we produce, in order
    with torch.no_grad():                               # still inference, still no gradients
        for _ in range(30):                             # produce 30 tokens; each iteration is one full forward pass over the whole sequence
            next_logits = model(cur).logits[0, -1]      # run the model, keep only example 0, LAST position: the scores for what comes next
            next_id = next_logits.argmax()              # greedy choice: the single highest-scoring vocabulary token
            generated.append(next_id.item())            # record it as a plain Python int
            cur = torch.cat([cur, next_id.view(1, 1)], dim=1)   # append it to the sequence (shape [1, len] -> [1, len+1]) and go round again
    out["hand_generated"] = tok.decode(generated)       # turn the 30 IDs back into readable text

    # 6. Same thing with the library's built-in generate()
    with torch.no_grad():                               # inference again
        full = model.generate(**inputs, max_new_tokens=60, do_sample=False)   # the library's loop: same predict-append-repeat, but with a KV cache so earlier tokens aren't recomputed; do_sample=False means greedy like ours
    out["library_generated"] = tok.decode(              # decode only the NEW tokens...
        full[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True   # ...by slicing off the prompt's length from the front, and dropping markers like <|im_end|>
    )

    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)   # highest VRAM used at any point: weights + temporary activations during generation
    return out                                          # ship the dict back to the laptop


@app.local_entrypoint()                                 # marks the function that runs on the Mac when you type `modal run`
def main():                                             # the remote control
    res = inspect_and_generate.remote("Explain what a GPU is in one sentence.")   # .remote() sends the call to a T4 machine and waits for the result dict
    for key, value in res.items():                      # loop over every result entry
        print(f"\n=== {key} ===\n{value}")              # print it with a header so each section is easy to find
