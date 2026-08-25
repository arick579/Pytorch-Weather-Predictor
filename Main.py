import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


import random
seed_value = 42

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


data_path = "pollution.csv"
if not os.path.exists(data_path):
    print("Downloading 5 years of weather data from UCI...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
    df = pd.read_csv(url)[['TEMP', 'DEWP', 'PRES']]
    df.to_csv(data_path, index=False)
else:
    print("Loading local weather dataset...")
    df = pd.read_csv(data_path)[['TEMP', 'DEWP', 'PRES']]

df.interpolate(method='linear', limit_direction='both', inplace=True)

data = df.values

train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]


scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


def create_sequences(data_array, seq_length):
    xs, ys = [], []
    for i in range(len(data_array) - seq_length):
        x = data_array[i:(i+seq_length)]
        y = data_array[i+seq_length][0]  # Target is TEMP (index 0)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Batching prevents Out-Of-Memory (OOM) crashes on large-scale datasets
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

# ==========================================
# HARDWARE ACCELERATION (NVIDIA CUDA)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nSystem Check - Training on device: {device}")

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :]) 
        return out

model = WeatherLSTM(input_size=3, hidden_size=16, num_layers=2).to(device)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 50 
patience = 5  # Allow the model 5 epochs of grace to try and improve before killing it
best_val_loss = float('inf')
epochs_no_improve = 0

train_losses, val_losses = [], []

print(f"\nInitiating Neural Network Training (Max {epochs} Epochs)...")

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    batch_train_loss = 0
    for X_batch, y_batch in train_loader:
        # Push batch data to the GPU
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_train_loss += loss.item()
    
    avg_train_loss = batch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- Validation Phase (Unseen Data) ---
    model.eval()
    batch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Push validation data to the GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            val_outputs = model(X_batch)
            val_loss = criterion(val_outputs, y_batch)
            batch_val_loss += val_loss.item()
            
    avg_val_loss = batch_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{epochs}] | Train MSE: {avg_train_loss:.4f} | Validation MSE: {avg_val_loss:.4f}')

    # --- EARLY STOPPING LOGIC ---
    # If the validation loss drops, reset the patience counter
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        # If the validation loss gets worse, tick the counter up
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\n🛑 EARLY STOPPING TRIGGERED! Validation loss failed to improve for {patience} consecutive epochs.")
            print(f"Optimal model achieved at Epoch {epoch + 1 - patience} with MSE: {best_val_loss:.4f}")
            break
