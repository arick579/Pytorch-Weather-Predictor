# Pytorch-Weather-Predictor

A deep-learning pipeline that utilizes a PyTorch Long Short-Term Memory (LSTM) neural network to analyze 5 years of historical telemetry and predict future weather patterns.

## The Tech Stack
* **Machine Learning:** PyTorch, Torch.nn
* **Data Processing:** Pandas, NumPy, Scikit-learn (StandardScaler)
* **Hardware Acceleration:** NVIDIA CUDA 
* **Visualization:** Matplotlib

## Core Architecture
1. **Fault-Tolerant Ingestion:** Automatically downloads massive datasets and utilizes linear interpolation to handle missing chronological telemetry without breaking the time-series sequence.
2. **Zero Data Leakage:** Implements strict sequential feature engineering, ensuring data scaling (`fit_transform`) is isolated entirely to the training set to prevent future data from influencing the model.
3. **Advanced Network Architecture:** Utilizes multi-layered LSTMs paired with a 20% Dropout rate to prevent the neural network from memorizing the training data.
4. **Automated Early Stopping:** Monitors validation loss on unseen data during the training loop, automatically halting execution to capture optimal model weights before catastrophic overfitting occurs.

## Visual Proof
<img width="1213" height="731" alt="image" src="https://github.com/user-attachments/assets/1cec766a-7e78-4d60-993d-26b9ddcf11c9" />

## How to Run Locally
**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```
**2. Execute the Training Pipeline:**
```bash
python Main.py
