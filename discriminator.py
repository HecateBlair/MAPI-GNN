import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
import random
from utils import set_seed, ConceptAutoEncoder

set_seed(37)


class MultiModalDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        # Defensive programming: Sort by Patient ID to ensure alignment
        data = data.sort_values(by=data.columns[0])

        self.features = data.iloc[:, 2:].values
        self.labels = data.iloc[:, 1].values
        self.patient_ids = data.iloc[:, 0].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label


class ConceptAutoEncoderTrainer:
    def __init__(self, autoencoder, model_dir="./saved_models"):
        self.autoencoder = autoencoder
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def get_last_encoder_linear_weight(self):
        linear_layers = [m for m in self.autoencoder.encoder.modules() if isinstance(m, nn.Linear)]
        return linear_layers[-1].weight if len(linear_layers) > 0 else None

    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=1e-3,
              regularization_strength_l1=0.0, orthogonal_constraint=False,
              lambda_orthogonality=1e-5, device="cpu", best_model_filename="best_model.pth"):

        self.autoencoder.to(device)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.SmoothL1Loss()
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.autoencoder.train()
            total_train_loss = 0
            for batch in train_loader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                _, outputs = self.autoencoder(inputs)
                loss = criterion(outputs, inputs)

                if regularization_strength_l1 > 0.0:
                    loss += regularization_strength_l1 * sum(
                        torch.sum(torch.abs(param)) for param in self.autoencoder.parameters())

                if orthogonal_constraint:
                    W = self.get_last_encoder_linear_weight()
                    if W is not None:
                        I = torch.eye(W.size(0), device=W.device)
                        loss += lambda_orthogonality * torch.norm(W @ W.t() - I)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            val_loss = self.evaluate(val_loader, criterion, device=device)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.autoencoder.state_dict(), os.path.join(self.model_dir, best_model_filename))

    def evaluate(self, data_loader, criterion, device="cpu"):
        self.autoencoder.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(device)
                _, outputs = self.autoencoder(inputs)
                loss = criterion(outputs, inputs)
                total_loss += loss.item()
        return total_loss / len(data_loader)


def generate_raw_data_tsne(features, labels, output_filename, scatter_cmap="Paired"):
    print("Generating t-SNE for original raw data...")
    tsne = TSNE(n_components=2, init='random', random_state=37)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=scatter_cmap, alpha=0.8,
                              s=40, edgecolors='k', linewidths=0.5)
        plt.legend(*scatter.legend_elements(), title="Label")
    else:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.8, c="blue", s=40, edgecolors='k',
                    linewidths=0.5)

    plt.title("t-SNE Visualization of Original Data")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"Visualization saved to {output_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Concept AutoEncoder")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input feature CSV')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save models and plots')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dataset = MultiModalDataset(args.csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    generate_raw_data_tsne(dataset.features, dataset.labels, os.path.join(args.save_dir, "tsne_original.pdf"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=37)
    encoding_layers = [256, 64, 24]
    input_dim = dataset.features.shape[1]

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.features, dataset.labels), 1):
        print(f"\nTraining Fold {fold}...")

        train_dataset = TensorDataset(torch.FloatTensor(dataset.features[train_idx]))
        val_dataset = TensorDataset(torch.FloatTensor(dataset.features[val_idx]))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        autoencoder = ConceptAutoEncoder(input_dim, encoding_layers).to(device)
        trainer = ConceptAutoEncoderTrainer(autoencoder, model_dir=args.save_dir)

        trainer.train(
            train_loader, val_loader,
            device=device,
            best_model_filename=f"best_ae_fold{fold}.pth",
            orthogonal_constraint=True
        )


if __name__ == "__main__":
    main()
