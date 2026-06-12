import torch
import time

print("=" * 50)
print("AMD GPU CHECK")
print("=" * 50)

print("Torch Version:", torch.__version__)
print("HIP Version:", torch.version.hip)
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():

    props = torch.cuda.get_device_properties(0)

    print("\nGPU Information")
    print("-" * 50)
    print("Architecture:", props.gcnArchName)
    print("Memory (GB):", round(props.total_memory / 1024**3))
    print("Compute Units:", props.multi_processor_count)

    print("\nRunning Matrix Multiplication Benchmark...")

    device = "cuda"

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    torch.cuda.synchronize()

    start = time.time()

    c = torch.matmul(a, b)

    torch.cuda.synchronize()

    elapsed = time.time() - start

    print("Result Device:", c.device)
    print("Execution Time:", round(elapsed, 4), "seconds")

else:
    print("No GPU detected.")