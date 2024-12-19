import numpy as np
import torch
import torch.nn as nn
import gc
import tracemalloc

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.dense(x)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Generate random data
    train = torch.rand(1000, 10, device=device)
    label = torch.rand(1000, device=device)

    tracemalloc.start()
    i = 0
    while i <= 100:
        i += 1
        
        optimizer.zero_grad()
        output = model(train)
        loss = criterion(output.squeeze(), label)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            del current, peak
            
        gc.collect()

if __name__ == "__main__":
    main()