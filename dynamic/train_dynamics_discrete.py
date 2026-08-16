import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
from lunar_discrete import LunarLanderDynamicsDiscrete

class DiscreteDatasetV2(Dataset):
    def __init__(self, data_dir):
        self.triplets = []
        files = glob.glob(f"{data_dir}/**/*.npz", recursive=True)
        print(f"Loading {len(files)} trajectory files...")
        for f in files:
            try:
                d = np.load(f, allow_pickle=False)
                states = d['lander_raw'].astype(np.float32)
                actions = d['action'].astype(np.int64)
                for i in range(len(actions) - 1):
                    self.triplets.append((
                        states[i],      # lander_raw[i]
                        states[i+1],    # lander_raw[i+1]
                        actions[i+1],   # action[i+1]
                        states[i+2]     # target: lander_raw[i+2]
                    ))
            except Exception as e:
                continue
        print(f"Total triplets: {len(self.triplets)}")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        s1, s2, a, s3 = self.triplets[idx]
        return (torch.tensor(s1),
                torch.tensor(s2),
                torch.tensor(a),
                torch.tensor(s3))

def main():
    data_dir = './data/lunar/lunar_discrete/train'
    save_dir = './checkpoints/dynamics_discrete'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = DiscreteDatasetV2(data_dir)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    model = LunarLanderDynamicsDiscrete(learnable=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for s1, s2, a, s3 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            s1 = s1.to(device)
            s2 = s2.to(device)
            a  = a.to(device)
            s3 = s3.to(device)

            optimizer.zero_grad()
            pred = model(s1, s2, a)
            loss = criterion(pred, s3)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f} | "
              f"main_power: {model.main_engine_power.item():.4f} | "
              f"side_power: {model.side_engine_power.item():.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'main_engine_power': model.main_engine_power.item(),
                'side_engine_power': model.side_engine_power.item(),
            }, f"{save_dir}/best.pt")

    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Learned main_engine_power: {model.main_engine_power.item():.4f}")
    print(f"Learned side_engine_power: {model.side_engine_power.item():.4f}")

if __name__ == '__main__':
    main()
