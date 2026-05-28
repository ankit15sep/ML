import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class BearingDataset(Dataset):
    def __init__(self, df_list, scalar, sequence_length=50):
        self.sequence_length = sequence_length
        self.samples = []
        self.scalar = scalar
        for df in df_list:
            df["kurtosis"] = df["kurtosis"].ewm(span=20, adjust=False).mean()
            df["rms"] = df["rms"].ewm(span=20, adjust=False).mean()
            df["shape_factor"] = df["shape_factor"].ewm(span=20, adjust=False).mean()
            df["pca_health_indicator"] = df["pca_health_indicator"].ewm(span=20, adjust=False).mean()
            # self.x = torch.tensor(df[["kurtosis", "rms", "shape_factor", "pca_health_indicator"]].values, dtype=torch.float32)
            # self.y = torch.tensor(df["rul"].values, dtype=torch.float32)
            x = df[["kurtosis", "rms", "shape_factor", "pca_health_indicator"]]
            y = torch.tensor(df["rul"].values, dtype=torch.float32)
            if self.scalar is not None:
                x = self.scalar.transform(x)
                x = torch.tensor(x, dtype=torch.float32)
            num_sequences = len(df) - sequence_length
            for i in range(num_sequences):
                self.samples.append((x[i:i + sequence_length], y[i + sequence_length]))

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the sample at index `idx` and return it as a tuple (input, target)
        return self.samples[idx]


class BearingLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # lstm_out will have the shape (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step (last hidden state) for prediction
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output


df_list = []
df_list_test = []
bearing_list_test1 = ["bearing_0_x", "bearing_1_x", "bearing_2_x", "bearing_3_x","bearing_0_y","bearing_1_y","bearing_2_y","bearing_3_y"]
# bearing_list_test1 = ["bearing_2_x", "bearing_3_x", "bearing_2_y","bearing_3_y"]
bearing_list_test2 = ["bearing_0_x", "bearing_1_x", "bearing_2_x", "bearing_3_x"]
# bearing_list_test3 = ["bearing_0_x", "bearing_1_x", "bearing_2_x", "bearing_3_x"]
bearing_list_test3 = ["bearing_2_x"]


test1 = pd.read_csv('data/processed/1st_test_features.csv')
test2 = pd.read_csv('data/processed/2nd_test_features.csv')
test3 = pd.read_csv('data/processed/3rd_test_features.csv')

# df['rms_smoothed'] = df['rms'].ewm(span=20, adjust=False).mean()
# df['kurtosis_smoothed'] = df['kurtosis'].ewm(span=20, adjust=False).mean()
#
# plt.plot(test1[test1['bearing'] == "bearing_3_x"]['pca_health_indicator'])
# plt.plot(test1[test1['bearing'] == "bearing_3_x"]['pca_health_indicator'].ewm(span=20, adjust=False).mean())
# plt.xlabel('Index')
# plt.ylabel('RUL')
# plt.title('Kurtosis vs RUL for Bearing 0_x')
# plt.show()

for bearing in bearing_list_test1:
    df_list.append(test1[test1['bearing'] == bearing].reset_index(drop=True))

for bearing in bearing_list_test2:
    df_list.append(test2[test2['bearing'] == bearing].reset_index(drop=True))

for bearing in bearing_list_test3:
    df_list_test.append(test3[test3['bearing'] == bearing].reset_index(drop=True))

all_data = pd.concat(df_list)
features = ["kurtosis", "rms", "shape_factor", "pca_health_indicator"]
print(all_data[features].head())
scalar = StandardScaler()
scalar.fit(all_data[features])
bearing_lstm = BearingLSTM()
dataset = BearingDataset(df_list, scalar)
# plt.figure()
# plt.plot(dataset.samples[0][0][:, 0].numpy(), label='kurtosis')
# plt.show()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
dataloader_test = DataLoader(BearingDataset(df_list_test, scalar), batch_size=32, shuffle=False)

# Example of iterating through the dataloader
for batch_x, batch_y in dataloader:
    print(batch_x.shape)  # Should be (batch_size, sequence_length, input_size)
    print(batch_y.shape)  # Should be (batch_size,)
    break

num_epochs = 100
input_size = 4
hidden_size = 64
num_layers = 2
model = BearingLSTM(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
loss_collected = []



for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    loss_collected.append(loss.item())
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}')

plt.figure()
plt.plot(loss_collected)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.grid()
plt.title('Training Loss Over Time')
plt.savefig('training_loss.png')
plt.close()


def predict(model, dataloader):
    model.eval()
    predictions = []
    rul = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            rul.extend(batch_y.cpu().numpy())
    return predictions, rul

predictions, rul = predict(model, dataloader_test)
plt.figure()
plt.plot(predictions, label='Predicted RUL')
plt.plot(rul, label='True RUL')
plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.title('Predicted vs True RUL')
plt.legend()
plt.grid()
plt.savefig('predicted_vs_true_rul.png')
plt.close()