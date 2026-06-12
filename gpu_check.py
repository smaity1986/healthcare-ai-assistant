import torch

print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("Available:", torch.cuda.is_available())
print("Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.current_device())


x = torch.tensor([1,2,3], device="cuda")
print(x)
print(x.device)

props = torch.cuda.get_device_properties(0)

print("Architecture:", props.gcnArchName)
print("Memory GB:", round(props.total_memory/1024**3))
print("Compute Units:", props.multi_processor_count)