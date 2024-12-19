import torch
import torch.nn as nn
import gc
import tracemalloc
from torch.profiler import profile, record_function, ProfilerActivity

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.dense(x)

def training_loop(model, optimizer, criterion, train, label, num_iterations):
    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(train)
        loss = criterion(output.squeeze(), label)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    train = torch.rand(1000, 10, device=device)
    label = torch.rand(1000, device=device)

    # Start tracemalloc before the loop
    tracemalloc.start()
    
    # Run training loop
    with profile(profile_memory=True) as prof:
        training_loop(model, optimizer, criterion, train, label, 100)
    
    # Stop tracemalloc after the loop
    tracemalloc.stop()
    print(prof.key_averages())

if __name__ == "__main__":
    main()