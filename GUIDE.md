# GPUs, PyTorch, CUDA and Open Source Models: Build-to-Learn Guide

Three small projects that teach how model training and inference actually work, from first principles:

1. **Project 1: A model learns from scratch.** One weight, one bias, the full training loop, no GPU needed.
2. **Project 2: CPU vs GPU on Modal.** Rent a GPU, move data to it, time it, measure VRAM.
3. **Project 3: Load an open source model on a Modal GPU.** Tokens, layers, parameter counts, memory, and text generation one token at a time.

---

## 0. Instructions for Claude Code (tutor mode)

> The learner is reading this file and has asked you to walk them through it. Follow these rules for the whole session.

**Who the learner is:** Comfortable with basic Python, but new to ML, PyTorch, CUDA and GPU hardware. Do not assume any ML background. Define every term the first time it appears.

**How to teach:**

1. **Go one step at a time.** Create or show one small chunk of code (a few lines), then stop. Do not dump a whole file and move on.
2. **Explain after every line of code.** For each line, say in plain language: what it does, *why it is there*, and what is physically happening (which numbers are being created, where they live: CPU RAM or GPU VRAM, which hardware does the work). Prefer small concrete numbers over abstractions.
3. **Make the learner predict before you run.** Before running any code, ask one question such as "what shape do you expect this weight table to be?" or "will `w.grad` be positive or negative?". Wait for the answer, correct gently, then run it and compare.
4. **Run the code for real** where you can (Project 1 locally; Projects 2 and 3 with `modal run`), and explain the actual output line by line.
5. **Check understanding every few steps** with a single focused question. When the learner explains something back correctly, say so and move on. Do not over-quiz.
6. **Use the "Key ideas to land" list** at the end of each project as your checklist. Do not finish a project until each item has come up.
7. **Do not write the README answers for the learner.** At the end of each project, ask them to write their own answers in their own words (the questions are in each project). Review what they wrote and correct anything wrong.
8. **If something errors,** treat it as a teaching moment: read the error with them, explain what it means, then fix it. Many of the errors listed in Section 7 are intentional lessons.
9. **Never invent API details.** Modal and Hugging Face APIs change. If a call fails, check the current docs instead of guessing.
10. **Never commit secrets.** No tokens or API keys in the repo. Modal's token lives in `~/.modal.toml`, outside the repo.

**Pacing:** Roughly 30 to 45 minutes per project. Offer a short recap at the end of each project and a final recap at the end of Project 3.

---

## 1. Reference: the mental model

### 1.1 What a model is

A model is a big pile of numbers (**weights**) plus a fixed formula that combines them with an input. **Training** means finding good values for the weights.

### 1.2 A node and a layer

A **node** (neuron) takes **all** of its inputs, multiplies each by its own private weight, adds them up, adds a **bias**, and optionally applies a simple cutoff (an **activation function**, e.g. ReLU: negative becomes 0). It outputs **one number**.

```
node output = relu( w1*x1 + w2*x2 + w3*x3 + bias )
```

- Sign of a weight: direction of influence on that node's output.
- Magnitude: how strongly. A weight of 0 means "ignore this input".

A **layer** is many nodes that all see the same inputs but have different weights. Stacking layers means the outputs of one layer are the inputs of the next. A layer with `n_in` inputs and `n_out` nodes has a weight table of shape `[n_out, n_in]` plus `n_out` biases.

Example: `nn.Linear(3, 100)` has a weight table of `[100, 3]` and 100 biases = 400 parameters. The next layer must be declared with 100 inputs: `nn.Linear(100, 1)` = 101 parameters.

### 1.3 The training loop (five steps)

```
repeat many times:
    1. take a batch of data
    2. forward pass: run it through the model to get predictions
    3. loss: measure how wrong the predictions are
    4. backward pass: compute a gradient for every weight
       (gradient = "if I increase this weight a little, does the loss go up or down, and how fast?")
    5. update: nudge every weight a small step opposite its gradient
```

Update rule: `weight = weight - learning_rate * gradient`.

### 1.4 Who does what: PyTorch vs CUDA vs GPU

```
Your training script (Python, runs on the CPU)
        |
PyTorch    tensors, layers, autograd, optimizers. Decides WHAT operation is needed.
        |
CUDA       Nvidia's software layer: driver + runtime + optimized libraries
           (cuBLAS for matrix multiply, cuDNN for neural net ops). Decides HOW to run it on the GPU.
        |
GPU        thousands of cores + its own memory (VRAM). Does the arithmetic.
```

- Restaurant analogy: PyTorch is the head chef calling out steps, CUDA is the kitchen equipment and shared procedures, the GPU is the large crew of cooks working in parallel.
- Same PyTorch code runs on CPU or GPU. Only the device of the tensors changes (`.to("cuda")`).
- Other hardware has equivalents: AMD uses ROCm, Apple Silicon uses Metal, Google TPUs use XLA.

### 1.5 Where things live

```
Disk              dataset, saved weights files
  | (load)
CPU RAM           your Python program, data being prepared
  | (PCIe link: slow, copy once with .to("cuda"))
GPU VRAM          weights, gradients, optimizer state, activations, current batch
  | (GPU memory bus: fast, used on every operation)
On-chip cache     small working set next to the cores
  |
Cores             the actual multiply-adds
```

- A core can only do math on numbers inside it, so every multiply reads operands from VRAM and writes results back.
- PyTorch can only combine tensors on the **same device**. Mixing CPU and GPU tensors raises an error.
- GPU work is **asynchronous**: Python queues it and moves on. To time GPU code correctly, call `torch.cuda.synchronize()` before stopping the clock.

### 1.6 Memory vs compute vs bandwidth (the kitchen)

| Resource | Kitchen analogy | What it limits |
|---|---|---|
| Memory capacity | Size of the warehouse | Does it fit at all? Otherwise: out-of-memory crash |
| Compute (FLOPs) | How fast the cooks chop | How long the math takes |
| Memory bandwidth | Width of the door between warehouse and kitchen | How fast data reaches the cores |

Whichever is smallest sets your speed.

- **Compute-bound:** lots of math per byte moved (large-batch training). The cores are the bottleneck.
- **Memory-bound (bandwidth-bound):** little math per byte moved (generating text one token at a time: every token reads all the weights but uses each only about once). The cores idle waiting for data. This is why HBM, very wide high-bandwidth memory, matters.

### 1.7 Rules of thumb

- Weights memory ~ `parameters x bytes per parameter` (fp32 = 4, fp16/bf16 = 2, int8 = 1, 4-bit = 0.5).
  Example: 7B parameters at 16-bit = ~14 GB.
- Training memory ~ weights + gradients + optimizer state + activations. Roughly 16 bytes per parameter for the first three with Adam, plus activations that grow with batch size and sequence length. Inference needs far less.
- Training compute ~ `6 x parameters x training tokens` FLOPs.
- Matrix multiply of two `n x n` matrices: ~`2 n^3` operations, but only `3 n^2` numbers of memory. Compute grows faster than memory, which is why bigger problems favor the GPU.
- Small workloads can be *slower* on a GPU: launch overhead and the RAM-to-VRAM copy dominate.

---

## 2. Setup

```bash
mkdir gpu-fundamentals && cd gpu-fundamentals
git init
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch modal
modal setup                                           # opens a browser to authenticate
printf ".venv/\n__pycache__/\n*.pyc\n" > .gitignore
```

Save this file in the repo as `GUIDE.md`.

Repo layout when finished:

```
gpu-fundamentals/
|-- GUIDE.md
|-- 01_training_loop.py        # Project 1 (local, no GPU)
|-- 02_cpu_vs_gpu.py           # Project 2 (Modal GPU)
|-- 03_open_source_model.py    # Project 3 (Modal GPU)
|-- README.md                  # learner's own results + answers
`-- .gitignore
```

---

## 3. Project 1: Watch a model learn

**Goal:** Teach one node to learn `y = 2x + 1` from four examples, and see every step of the training loop with nothing hidden.

**File:** `01_training_loop.py` (run with `python 01_training_loop.py`)

```python
import torch

# Data: inputs and the correct answers
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])       # y = 2x + 1

# The "model": one weight and one bias, starting at 0
w = torch.tensor([[0.0]], requires_grad=True)        # requires_grad: "track this for autograd"
b = torch.tensor([0.0], requires_grad=True)

for step in range(200):
    pred = x @ w + b                    # forward pass
    loss = ((pred - y) ** 2).mean()     # how wrong are we? (mean squared error)
    loss.backward()                     # backward pass: fills w.grad and b.grad
    with torch.no_grad():               # updates should not be tracked by autograd
        w -= 0.01 * w.grad              # update: step opposite the gradient
        b -= 0.01 * b.grad
        w.grad.zero_()                  # clear gradients, or they accumulate
        b.grad.zero_()
    if step % 40 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")
```

**Prediction questions (ask before running):**
- What will the loss be at step 0? (Hint: predictions are all 0.)
- Which way will `w` move first, up or down?

**Experiments:**
1. Print `w.grad` right after `loss.backward()`. When `w` is below its target, is the gradient positive or negative? What does the update do with that sign?
2. Change the learning rate to `0.5`, then to `0.0001`. Predict first. (Expect `0.5` to blow up and `0.0001` to learn very slowly.)
3. Replace the hand-written update with PyTorch's optimizer: create `opt = torch.optim.SGD([w, b], lr=0.01)` and use `opt.step()` and `opt.zero_grad()`. Confirm it behaves the same.
4. Turn it into a real layer: replace `w`, `b` with `model = torch.nn.Linear(1, 1)` and print `model.weight.shape` and `model.bias.shape`. Then `list(model.parameters())`.

**Key ideas to land:**
- Weights are the learned dials; here there are exactly two.
- The five-step loop: forward, loss, backward, update, repeat.
- A gradient's sign tells you which direction to turn a dial; its size tells you how much that dial matters right now.
- Autograd records the operations (the "receipt") during the forward pass and walks it backwards on `loss.backward()`.
- Gradients accumulate unless zeroed.
- Learning rate: too big diverges, too small crawls.
- `nn.Linear(1, 1)` is exactly this `w` and `b`, packaged.

**README questions (learner answers in their own words):**
1. Which way did `w.grad` point while `w` was below 2, and what did the update do with it?
2. What happened with learning rate 0.5 and with 0.0001, and why?
3. In your own words: what is a weight, and what does training change?

---

## 4. Project 2: CPU vs GPU on Modal

**Goal:** Rent a real GPU, copy data from CPU RAM to GPU VRAM, time the same matrix multiply on both, and verify the memory math.

**File:** `02_cpu_vs_gpu.py` (run with `modal run 02_cpu_vs_gpu.py`)

```python
import modal

app = modal.App("gpu-fundamentals")
image = modal.Image.debian_slim().pip_install("torch")


@app.function(image=image, gpu="T4", cpu=4, memory=4096)
def compare(n: int):
    import time
    import torch

    torch.zeros(1, device="cuda")            # warm-up: initialize CUDA so setup time isn't measured

    a = torch.randn(n, n)                    # lives in CPU RAM
    b = torch.randn(n, n)

    t = time.time()
    a @ b                                    # matrix multiply on the CPU
    cpu_s = time.time() - t

    t = time.time()
    a_gpu = a.to("cuda")                     # copy RAM -> VRAM over PCIe
    b_gpu = b.to("cuda")
    torch.cuda.synchronize()                 # wait until the copy has really finished
    copy_s = time.time() - t

    _ = a_gpu @ b_gpu                        # warm-up multiply (first call loads GPU libraries)
    torch.cuda.synchronize()

    t = time.time()
    a_gpu @ b_gpu                            # the timed GPU multiply
    torch.cuda.synchronize()                 # wait for the GPU, which runs asynchronously
    gpu_s = time.time() - t

    return {
        "n": n,
        "cpu_s": round(cpu_s, 4),
        "copy_to_gpu_s": round(copy_s, 4),
        "gpu_s": round(gpu_s, 4),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "gpu": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint()
def main():
    for n in [500, 1000, 2000, 4000, 8000]:
        print(compare.remote(n))
```

**Prediction questions (ask before running):**
- At which size `n` do you think the GPU starts beating the CPU? Why might it lose at small sizes?
- How many bytes is one `n x n` float32 matrix? What should `vram_gb` be at the end? (Three matrices are alive: `a_gpu`, `b_gpu`, and `_`.)

**Experiments:**
1. Check the memory math: expected VRAM = `3 x n^2 x 4 bytes`. At `n = 4000` that is about 0.19 GB. Does `vram_gb` match?
2. Look at how compute and memory scale: going from `n = 4000` to `8000` multiplies memory by about 4 but compute by about 8. Do the timings roughly agree?
3. Remove the `torch.cuda.synchronize()` just before the final `gpu_s` measurement. The GPU time will look absurdly small. Explain why (asynchronous queueing).
4. Compare `copy_to_gpu_s` with `gpu_s`. What does that say about the cost of crossing between RAM and VRAM?
5. Break it on purpose: try `a @ b_gpu` (one CPU tensor, one GPU tensor) and read the error.
6. Try `dtype=torch.float16` for the GPU tensors (`a.to("cuda", dtype=torch.float16)`). What happens to VRAM and to speed?

**Key ideas to land:**
- Same math, different device; the difference is parallelism.
- `.to("cuda")` physically copies data across the PCIe link into VRAM. Do it once, then keep data on the GPU.
- Tensors on different devices cannot be combined.
- GPU calls are asynchronous; correct timing needs `synchronize()`.
- Small jobs can lose on GPU because of copy and launch overhead.
- VRAM usage is predictable arithmetic: elements x bytes per element.
- Modal: a decorated Python function runs remotely on a rented GPU; `.remote()` calls it from the laptop; billing is by use.

**README questions:**
1. At what size did the GPU start beating the CPU, and why not at smaller sizes?
2. Did `vram_gb` match your prediction? Show the arithmetic.
3. What did removing `synchronize()` show you?
4. How does copy time compare to multiply time, and what does that imply?

---

## 5. Project 3: Load an open source model

**Goal:** Load a real open source language model onto a Modal GPU, look inside it, count its parameters, check its memory against the rule of thumb, and watch it generate text one token at a time.

We use `Qwen/Qwen2.5-0.5B-Instruct`: about half a billion parameters, public (no access token needed), small enough for a cheap GPU.

**File:** `03_open_source_model.py` (run with `modal run 03_open_source_model.py`)

```python
import modal

app = modal.App("open-source-model")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@app.function(image=image, gpu="T4", memory=8192, timeout=900)
def inspect_and_generate(prompt: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out = {}

    # 1. Tokenizer: text -> list of integer token IDs, and back
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ids = tok(prompt)["input_ids"]
    out["token_ids"] = ids
    out["tokens"] = tok.convert_ids_to_tokens(ids)

    # 2. Model: download the weights and build the architecture, in 16-bit
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to("cuda")                 # copy all weights from CPU RAM into GPU VRAM
    model.eval()                             # inference mode

    # 3. Look inside: the layers and the parameter count
    out["architecture"] = str(model)
    n_params = sum(p.numel() for p in model.parameters())
    out["n_params"] = n_params
    out["predicted_weights_gb"] = round(n_params * 2 / 1e9, 3)    # 2 bytes each in fp16
    out["actual_vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    out["first_weight_name_and_shape"] = [
        (name, tuple(p.shape)) for name, p in list(model.named_parameters())[:4]
    ]

    # 4. One forward pass: what does the model output?
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")      # input tokens: RAM -> VRAM
    with torch.no_grad():
        logits = model(**inputs).logits                     # [batch, seq_len, vocab_size]
    out["logits_shape"] = tuple(logits.shape)

    # 5. Generate one token at a time, by hand
    cur = inputs["input_ids"]
    generated = []
    with torch.no_grad():
        for _ in range(30):
            next_logits = model(cur).logits[0, -1]          # scores for the NEXT token only
            next_id = next_logits.argmax()                  # pick the highest-scoring token
            generated.append(next_id.item())
            cur = torch.cat([cur, next_id.view(1, 1)], dim=1)   # append it and repeat
    out["hand_generated"] = tok.decode(generated)

    # 6. Same thing with the library's built-in generate()
    with torch.no_grad():
        full = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    out["library_generated"] = tok.decode(
        full[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return out


@app.local_entrypoint()
def main():
    res = inspect_and_generate.remote("Explain what a GPU is in one sentence.")
    for key, value in res.items():
        print(f"\n=== {key} ===\n{value}")
```

> Note: newer versions of `transformers` may print a deprecation warning suggesting `dtype=` instead of `torch_dtype=`. If so, switch the argument name. The meaning is the same.

**Prediction questions (ask before running):**
- Roughly how many parameters do you expect, and how many GB should the 16-bit weights take? (Use `params x 2 bytes`.)
- The output `logits` has shape `[1, seq_len, vocab_size]`. What do you think each of the three dimensions means?

**Walkthrough order for the tutor:**
1. **Tokens.** Print `tokens` and `token_ids`. Text becomes integers; the model never sees letters. Each ID will be looked up in an **embedding table** (a weight table of shape `[vocab_size, hidden_size]`) to become a list of numbers. This is the bridge from words to the matrix math from Project 1.
2. **Architecture.** Read the printed `architecture` together. Find the `Linear` layers and the embedding. Point out that each `Linear(in, out)` is the same weight table as in Project 1, just much bigger. Mention attention (each token looks at other tokens) as a new structure built from the same multiply-and-add.
3. **Parameter count and memory.** Compare `predicted_weights_gb` with `actual_vram_gb`. They should be close; explain any gap (e.g. buffers, rounding of the count).
4. **Weight names and shapes.** Look at `first_weight_name_and_shape`. Explain that a weights file is literally a dictionary of names to number tables, and that loading a model means building the architecture and filling each named slot.
5. **Logits.** The last position's logits are one score per vocabulary word. The highest score is the model's guess for the next token.
6. **Generation.** The hand-written loop is the whole idea: predict one token, append it, repeat. Explain why this is **memory-bandwidth-bound**: every new token reads all the weights from VRAM but does only about two operations per weight. (Optional: mention the KV cache, which our hand loop does not use, so it recomputes earlier tokens every step. That is why `generate()` is faster.)
7. **Compare** `hand_generated` and `library_generated`. They should start the same (both are greedy: always the top-scoring token).

**Experiments:**
1. Change `torch.float16` to `torch.float32` and compare `actual_vram_gb`. It should about double.
2. Count how many parameters live in the embedding table vs the rest. (`model.get_input_embeddings().weight.numel()`.)
3. Use a longer prompt and watch how `logits_shape` changes in the sequence dimension only.
4. Try a larger Qwen variant (for example the 1.5B instruct model) and predict the VRAM first. Check that it still fits on a T4 (16 GB).
5. Swap `argmax()` for sampling from the softmax of the logits (`torch.multinomial(torch.softmax(next_logits.float(), -1), 1)`). Notice the output now varies between runs.

**Key ideas to land:**
- An open source model is an **architecture** (code) plus **weights** (a file of named number tables), and both must match.
- Text is tokenized into IDs, embedded into vectors, pushed through many layers of matrix multiplies, and turned into a score for every possible next token.
- Parameter count x bytes per parameter predicts the VRAM you need just to hold the weights.
- Generation is a loop of one forward pass per token.
- Generation speed is dominated by memory bandwidth, not raw compute.
- Precision (fp32, fp16, 4-bit) is a memory-for-accuracy tradeoff; quantization shrinks weights further.
- Fine-tuning would be Project 1's loop applied to these same weights.

**README questions:**
1. How close was your predicted weight memory to the actual VRAM? Show the arithmetic.
2. What do the three dimensions of `logits_shape` mean?
3. Explain in your own words how the model generates a sentence.
4. Why is generating text memory-bandwidth-bound while training with a big batch is compute-bound?

---

## 6. Publish to GitHub

Write `README.md` with:
- One paragraph on what the repo is.
- For each project: how to run it, your real output pasted in, and your own answers to that project's README questions.
- A short "what I learned" section.

Then:

```bash
git add .
git commit -m "GPU fundamentals: training loop, CPU vs GPU, open source model on Modal"
gh repo create gpu-fundamentals --public --source=. --push
# or create the repo on github.com, then: git remote add origin <url> && git push -u origin main
```

Before pushing, check that no tokens or secrets are in the repo (`git status`, and grep for "token" or "key").

---

## 7. Troubleshooting and intentional lessons

| Symptom | Meaning |
|---|---|
| `Expected all tensors to be on the same device` | Model and data are on different devices. Move both with `.to(device)`. |
| `mat1 and mat2 shapes cannot be multiplied` | Layer input size does not match the previous layer's output size. |
| `CUDA out of memory` | The data does not fit in VRAM. Reduce size/batch, use lower precision, or pick a bigger GPU. |
| GPU time looks impossibly small | Missing `torch.cuda.synchronize()`; you timed the queueing, not the work. |
| GPU slower than CPU at small sizes | Launch and copy overhead dominate tiny workloads. Expected. |
| Loss becomes `nan` or `inf` | Learning rate too large; the updates overshoot and diverge. |
| `w.grad` is `None` | The tensor was not created with `requires_grad=True`, or was modified outside autograd tracking. |
| First Modal run is slow | Modal is building the container image and (in Project 3) downloading the model. Later runs reuse the cached image. |
| `torch_dtype` deprecation warning | Rename the argument to `dtype`. |
| Modal API call fails unexpectedly | APIs change. Check the current Modal and Hugging Face docs rather than guessing. |

---

## 8. Cheat sheet

```python
x.to("cuda")                                 # copy a tensor RAM -> VRAM
x.to("cpu")                                  # copy back
torch.cuda.is_available()                    # is a GPU visible?
torch.cuda.synchronize()                     # wait for queued GPU work
torch.cuda.memory_allocated()                # bytes currently held on the GPU
torch.cuda.max_memory_allocated()            # peak bytes
sum(p.numel() for p in model.parameters())   # number of weights
model.named_parameters()                     # (name, tensor) for every weight table
loss.backward()                              # compute gradients
opt.step(); opt.zero_grad()                  # update weights; clear gradients
with torch.no_grad(): ...                    # don't track gradients (inference)
```

| Term | One-line meaning |
|---|---|
| Weight | A learned number; one dial |
| Bias | A learned offset added in each node |
| Activation | A cutoff function (e.g. ReLU) applied after the sum |
| Parameter | Any learned number (weights + biases) |
| Forward pass | Run input through the model to get a prediction |
| Loss | A single number for how wrong the prediction is |
| Gradient | How the loss changes if one weight changes |
| Backward pass | Compute all gradients (autograd) |
| Optimizer | The rule that updates weights from gradients |
| Epoch / batch | One pass over the data / a chunk of data processed together |
| Tensor | A multi-dimensional array that can live on CPU or GPU |
| VRAM | The GPU's own memory |
| CUDA | Nvidia's software layer for running work on its GPUs |
| Kernel | A function that runs on the GPU across many threads |
| Token | A chunk of text (word piece) mapped to an integer ID |
| Embedding | The weight-table row that turns a token ID into a vector |
| Logits | Raw scores, one per vocabulary token, for the next token |
| Inference | Using a trained model |
| Fine-tuning | Continuing training from existing weights on new data |
| Quantization | Storing weights in fewer bits to save memory |
| HBM | Very high-bandwidth stacked memory used by data center GPUs |
| Compute-bound / memory-bound | Limited by arithmetic speed / by data movement speed |
