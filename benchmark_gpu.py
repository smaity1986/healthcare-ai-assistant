import torch
import time

device = "cuda"

print("Warmup...")

for _ in range(5):
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    c = torch.matmul(a, b)

torch.cuda.synchronize()

print("Benchmarking...")

times = []

for i in range(10):

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    torch.cuda.synchronize()

    start = time.perf_counter()

    c = torch.matmul(a, b)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    times.append(elapsed)

    print(f"Run {i+1}: {elapsed:.4f}s")

print("\nAverage:", sum(times)/len(times))