import torch
import torch.nn as nn
import numpy as np
import dgl
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import random
import argparse
from utils import set_seed, ConceptAutoEncoder

set_seed(42)


def analyze_significant_features(concept_autoencoder, feature, device, percentage=0.05):
    """
    Identify significant features by perturbing input features and measuring latent space impact.
    """
    feature = torch.FloatTensor(feature).reshape(1, -1).to(device)
    feature_dim = feature.shape[1]

    latent_dim = None
    for module in concept_autoencoder.encoder:
        if isinstance(module, nn.Linear):
            latent_dim = module.out_features

    significant_features = [[] for _ in range(latent_dim)]
    feature_impacts = [[] for _ in range(latent_dim)]

    with torch.no_grad():
        original_latent = concept_autoencoder.encoder(feature)

    for dim in range(latent_dim):
        impacts = []
        for feature_idx in range(feature_dim):
            perturbed = feature.clone()
            perturbed[0, feature_idx] = 0

            with torch.no_grad():
                perturbed_latent = concept_autoencoder.encoder(perturbed)

            impact = torch.abs(perturbed_latent[0, dim] - original_latent[0, dim]).item()
            impacts.append((impact, feature_idx))

        sorted_features = sorted(impacts, key=lambda x: x[0], reverse=True)
        num_significant = max(1, int(feature_dim * percentage))
        significant_features[dim] = [f[1] for f in sorted_features[:num_significant]]

        max_impact = sorted_features[0][0]
        min_impact = sorted_features[-1][0]
        if max_impact > min_impact:
            norm_impacts = [(impact - min_impact) / (max_impact - min_impact)
                            for impact, _ in sorted_features[:num_significant]]
        else:
            norm_impacts = [1.0] * num_significant
        feature_impacts[dim] = norm_impacts

    return significant_features, feature_impacts


def build_graphs(feature, significant_features, feature_impacts, latent_dim, k_neighbors=5):
    """Construct activation plane graphs based on significant features."""
    graphs = []
    for dim in range(latent_dim):
        g = dgl.graph([])
        g.add_nodes(len(feature))
        g.ndata['feat'] = torch.FloatTensor(feature).reshape(-1, 1)

        sig_features = significant_features[dim]
        impacts = feature_impacts[dim]

        feature_matrix = np.array([feature[sig] for sig in sig_features]).reshape(-1, 1)

        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        nbrs.fit(feature_matrix)
        distances, indices = nbrs.kneighbors(feature_matrix)

        src, dst, weights = [], [], []
        for i in range(len(sig_features)):
            for j in range(k_neighbors):
                neighbor_idx = indices[i, j]
                if neighbor_idx != i:
                    src.append(sig_features[i])
                    dst.append(sig_features[neighbor_idx])
                    weights.append((impacts[i] + impacts[neighbor_idx]) / 2)

        if src:
            g.add_edges(src, dst)
            g.edata['weight'] = torch.FloatTensor(weights).reshape(-1, 1)

        graphs.append(g)
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Input features CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained ConceptAutoEncoder')
    parser.add_argument('--save_dir', type=str, default='./graphs', help='Output directory for graphs')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    data = pd.read_csv(args.csv_path)
    # Sort data to ensure alignment
    data = data.sort_values(by=data.columns[0])

    features = data.iloc[:, 2:].values
    labels = data.iloc[:, 1].values
    patient_ids = data.iloc[:, 0].values

    # Must match training configuration
    encoding_layers = [256, 64, 24]
    input_dim = features.shape[1]

    concept_autoencoder = ConceptAutoEncoder(input_dim, encoding_layers).to(device)
    concept_autoencoder.load_state_dict(torch.load(args.model_path, map_location=device))
    concept_autoencoder.eval()

    for idx in tqdm(range(len(features)), desc="Constructing Graphs"):
        feature = features[idx]
        label = labels[idx]
        patient_id = patient_ids[idx]

        significant_features, feature_impacts = analyze_significant_features(
            concept_autoencoder, feature, device, percentage=0.1
        )

        graphs = build_graphs(
            feature, significant_features, feature_impacts,
            latent_dim=encoding_layers[-1], k_neighbors=5
        )

        save_path = os.path.join(args.save_dir, f"{int(label)}", f"{patient_id}", "graphs.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(graphs, save_path)


if __name__ == "__main__":
    main()

