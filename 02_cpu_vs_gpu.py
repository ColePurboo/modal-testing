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
