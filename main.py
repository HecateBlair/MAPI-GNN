import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import dgl
from dgl.nn import GraphConv, GATConv
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import random
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import set_seed
import json
import time
import matplotlib.pyplot as plt


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths (will be updated via argparse)
    graphs_dir = "."
    csv_path = "."

    random_state = 42
    n_splits = 5
    use_balanced_sampling = False

    # Model Hyperparameters
    plane_in_dim = 1
    plane_hidden_dim = 64
    plane_out_dim = 32
    num_planes = 24
    num_heads = 2
    node_hidden_dim = 128
    num_classes = 2
    k_neighbors = 5

    # Training Hyperparameters
    num_epochs = 100
    batch_size = 8
    patience = 15
    learning_rate = 0.001
    weight_decay = 1e-3
    feat_dropout = 0.4
    gnn_dropout = 0.4

    lambda_reconstruction = 0.3
    lambda_classification = 1.0


set_seed(Config.random_state)


class PlaneGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=Config.num_heads):
        super(PlaneGraphEncoder, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, num_heads, feat_drop=Config.feat_dropout, attn_drop=Config.feat_dropout,
                            residual=True, allow_zero_in_degree=True)
        self.gat2 = GATConv(hidden_dim * num_heads, out_dim, 1, feat_drop=Config.feat_dropout,
                            attn_drop=Config.feat_dropout, residual=True, allow_zero_in_degree=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim * num_heads),
            nn.BatchNorm1d(hidden_dim * num_heads),
            nn.ReLU(),
            nn.Dropout(Config.feat_dropout),
            nn.Linear(hidden_dim * num_heads, in_dim)
        )

    def forward(self, g, return_reconstruction=True):
        h = g.ndata['feat']
        if h.dim() == 3: h = h.squeeze(1)
        h = self.gat1(g, h)
        h = h.reshape(h.shape[0], -1)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=Config.feat_dropout, training=self.training)
        h = self.gat2(g, h)
        h = h.squeeze(1)
        h = self.bn2(h)
        h = F.relu(h)
        g.ndata['h'] = h
        graph_rep = dgl.mean_nodes(g, 'h')
        if return_reconstruction:
            reconstruction = self.decoder(h)
            return graph_rep, reconstruction
        return graph_rep


class MaskedGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(MaskedGraphConv, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, norm='both', allow_zero_in_degree=True)

    def forward(self, g, h, mask):
        with g.local_scope():
            device = g.device
            src, dst = g.edges()
            mask = mask.to(device)
            edge_mask = (mask[src] & mask[dst]).to(device)
            original_weights = g.edata.get('weight', None)
            g.edata['_masked_weight'] = edge_mask.float().unsqueeze(-1).to(device)
            if original_weights is not None:
                g.edata['_masked_weight'] = g.edata['_masked_weight'] * original_weights
            return self.conv(g, h, edge_weight=g.edata['_masked_weight'])


class End2EndModel(nn.Module):
    def __init__(self, input_dim, original_feature_dim):
        super(End2EndModel, self).__init__()
        self.plane_encoders = nn.ModuleList(
            [PlaneGraphEncoder(Config.plane_in_dim, Config.plane_hidden_dim, Config.plane_out_dim) for _ in
             range(Config.num_planes)])
        self.fused_dim = original_feature_dim + Config.num_planes * Config.plane_out_dim
        self.feature_transform = nn.Sequential(nn.Linear(self.fused_dim, Config.node_hidden_dim),
                                               nn.BatchNorm1d(Config.node_hidden_dim), nn.ReLU(),
                                               nn.Dropout(Config.gnn_dropout))
        self.node_gnn = nn.ModuleList(
            [MaskedGraphConv(Config.node_hidden_dim, Config.node_hidden_dim) for _ in range(3)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(Config.node_hidden_dim) for _ in range(3)])
        self.classifier = nn.Sequential(nn.Linear(Config.node_hidden_dim, Config.node_hidden_dim // 2),
                                        nn.LayerNorm(Config.node_hidden_dim // 2), nn.ReLU(),
                                        nn.Dropout(Config.gnn_dropout),
                                        nn.Linear(Config.node_hidden_dim // 2, Config.num_classes))

    def forward(self, patient_graphs_list, patient_graph, original_features, mask):
        batch_size = len(patient_graphs_list)
        total_reconstruction_loss = 0
        all_patient_features = []
        for patient_graphs in patient_graphs_list:
            patient_plane_features = []
            for plane_idx, plane_graph in enumerate(patient_graphs):
                graph_rep, reconstruction = self.plane_encoders[plane_idx](plane_graph)
                original = plane_graph.ndata['feat']
                if original.dim() == 3: original = original.squeeze(1)
                if reconstruction.dim() == 3: reconstruction = reconstruction.squeeze(1)
                total_reconstruction_loss += F.mse_loss(reconstruction, original)
                patient_plane_features.append(graph_rep)
            all_patient_features.append(torch.cat(patient_plane_features, dim=0))

        all_plane_features = torch.stack(all_patient_features).view(batch_size, -1)
        node_features = torch.cat([original_features, all_plane_features], dim=1)
        h = self.feature_transform(node_features)
        h_list = [h]
        for gnn, bn in zip(self.node_gnn, self.bn_layers):
            h_new = gnn(patient_graph, h, mask)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=Config.gnn_dropout, training=self.training)
            h = h_new + h
            h_list.append(h)
        h = torch.stack(h_list, dim=0).mean(dim=0)
        logits = self.classifier(h)
        return logits, total_reconstruction_loss / (batch_size * Config.num_planes)


class PatientDataset:
    def __init__(self, csv_path, graphs_dir):
        print("\n==== Loading dataset ====")
        self.df = pd.read_csv(csv_path)
        # Defensive programming: Sort by Patient ID
        self.df = self.df.sort_values(by=self.df.columns[0])
        self.patient_ids = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1].values
        self.original_features = self.df.iloc[:, 2:].values
        self.graphs_dir = graphs_dir
        self.original_feature_dim = self.original_features.shape[1]
        print("Loading saved planar graphs...")
        self.patient_graphs = self.load_saved_graphs()
        print("Building fusion-relation graph...")
        self.relation_graph = self.build_relation_graph()

    def load_saved_graphs(self):
        patient_graphs = []
        for patient_id, label in zip(self.patient_ids, self.labels):
            graph_path = os.path.join(self.graphs_dir, f"{int(label)}", f"{patient_id}", "graphs.pt")
            try:
                graphs = torch.load(graph_path)
                patient_graphs.append(graphs)
            except Exception as e:
                raise RuntimeError(f"Error loading graph for {patient_id}: {str(e)}")
        return patient_graphs

    def build_relation_graph(self):
        features_tensor = torch.FloatTensor(self.original_features)
        sim_matrix = torch.cdist(features_tensor, features_tensor)
        _, indices = torch.topk(sim_matrix, k=Config.k_neighbors + 1, dim=1, largest=False)
        src, dst = [], []
        for i in range(len(self.patient_ids)):
            for j in indices[i][1:]:
                src.extend([i, j])
                dst.extend([j, i])
        g = dgl.graph((torch.tensor(src), torch.tensor(dst)))
        g.ndata['feat'] = features_tensor
        g.ndata['label'] = torch.LongTensor(self.labels)
        return g


def train_model(model, dataset, train_mask, val_mask, device, fold_dir):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    patient_graph = dataset.relation_graph.to(device)
    original_features = torch.FloatTensor(dataset.original_features).to(device)
    labels = torch.LongTensor(dataset.labels).to(device)
    train_mask, val_mask = train_mask.to(device), val_mask.to(device)
    patient_graphs_list = [[g.to(device) for g in patient_graphs] for patient_graphs in dataset.patient_graphs]

    best_val_acc = 0
    no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(Config.num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, recon_loss = model(patient_graphs_list, patient_graph, original_features, train_mask)
        clf_loss = criterion(logits[train_mask], labels[train_mask])
        loss = Config.lambda_classification * clf_loss + Config.lambda_reconstruction * recon_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, train_pred = torch.max(logits[train_mask], dim=1)
        train_acc = (train_pred == labels[train_mask]).float().mean()

        model.eval()
        with torch.no_grad():
            val_logits, val_recon_loss = model(patient_graphs_list, patient_graph, original_features, val_mask)
            val_clf_loss = criterion(val_logits[val_mask], labels[val_mask])
            val_loss = Config.lambda_classification * val_clf_loss + Config.lambda_reconstruction * val_recon_loss
            _, val_pred = torch.max(val_logits[val_mask], dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean()

        scheduler.step(val_acc)
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc.item())

        print(
            f'Epoch {epoch:03d}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
        else:
            no_improve += 1
            if no_improve >= Config.patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break
    return history, best_val_acc.item()


def evaluate(model, dataset, mask, device, save_predictions=False, fold=None, save_dir=None):
    model.eval()
    with torch.no_grad():
        patient_graph = dataset.relation_graph.to(device)
        original_features = torch.FloatTensor(dataset.original_features).to(device)
        labels = torch.LongTensor(dataset.labels).to(device)
        mask = mask.to(device)
        patient_graphs_list = [[g.to(device) for g in patient_graphs] for patient_graphs in dataset.patient_graphs]
        logits, _ = model(patient_graphs_list, patient_graph, original_features, mask)
        probs = torch.softmax(logits[mask], dim=1)
        _, preds = torch.max(logits[mask], dim=1)
        y_true, y_pred, y_prob = labels[mask].cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()

        if save_predictions and fold is not None and save_dir is not None:
            mask_cpu = mask.cpu()
            patient_ids = dataset.patient_ids[mask_cpu.numpy()]
            results_df = pd.DataFrame({'patient_id': patient_ids, 'true_label': y_true, 'predicted_label': y_pred,
                                       'prob_class_0': y_prob[:, 0], 'prob_class_1': y_prob[:, 1]})
            results_df.to_csv(os.path.join(save_dir, f'fold_{fold}_predictions.csv'), index=False)

        metrics = {'acc': (y_pred == y_true).mean(), 'auc': roc_auc_score(y_true, y_prob[:, 1]),
                   'precision': precision_score(y_true, y_pred), 'recall': recall_score(y_true, y_pred),
                   'f1': f1_score(y_true, y_pred)}
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp)
        return metrics


def create_nested_cv_folds(dataset, n_splits=5, random_state=37, use_balanced=True):
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter
    labels, num_nodes = dataset.labels, len(dataset.patient_ids)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_folds = []
    fold_assignments = np.zeros(num_nodes)
    for fold_idx, (_, fold_idx_test) in enumerate(outer_cv.split(np.zeros(num_nodes), labels), 1):
        fold_assignments[fold_idx_test] = fold_idx

    for test_fold in range(1, n_splits + 1):
        test_idx = np.where(fold_assignments == test_fold)[0]
        val_fold = test_fold % n_splits + 1
        if val_fold == test_fold: val_fold = val_fold % n_splits + 1
        val_idx = np.where(fold_assignments == val_fold)[0]
        train_idx = np.where((fold_assignments != test_fold) & (fold_assignments != val_fold))[0]

        if use_balanced:
            train_labels = labels[train_idx]
            class_counts = Counter(train_labels)
            min_class_count = min(class_counts.values())
            balanced_train_indices = []
            for label in np.unique(labels):
                label_indices = train_idx[train_labels == label]
                selected_indices = np.random.RandomState(random_state).choice(label_indices, size=min_class_count,
                                                                              replace=False)
                balanced_train_indices.extend(selected_indices)
            balanced_train_indices = np.array(balanced_train_indices)
            np.random.RandomState(random_state).shuffle(balanced_train_indices)
            train_idx = balanced_train_indices
        all_folds.append((train_idx, val_idx, test_idx))
    return all_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--graphs_dir', type=str, required=True)
    args = parser.parse_args()

    Config.csv_path = args.csv_path
    Config.graphs_dir = args.graphs_dir
    print(f"Using device: {Config.device}")

    dataset = PatientDataset(Config.csv_path, Config.graphs_dir)
    results_dir = os.path.join(Config.graphs_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    folds = create_nested_cv_folds(dataset, n_splits=Config.n_splits, random_state=Config.random_state,
                                   use_balanced=Config.use_balanced_sampling)
    cv_results = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(folds, 1):
        print(f"\n==== Starting Fold {fold}/{Config.n_splits} ====")
        fold_dir = os.path.join(results_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        train_mask = torch.zeros(len(dataset.patient_ids), dtype=torch.bool).to(Config.device)
        val_mask = torch.zeros(len(dataset.patient_ids), dtype=torch.bool).to(Config.device)
        test_mask = torch.zeros(len(dataset.patient_ids), dtype=torch.bool).to(Config.device)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        model = End2EndModel(input_dim=dataset.original_features.shape[1],
                             original_feature_dim=dataset.original_features.shape[1]).to(Config.device)
        history, best_val_acc = train_model(model, dataset, train_mask, val_mask, Config.device, fold_dir)

        best_model_path = os.path.join(fold_dir, 'best_model.pt')
        model.load_state_dict(torch.load(best_model_path))
        test_metrics = evaluate(model, dataset, test_mask, Config.device, save_predictions=True, fold=fold,
                                save_dir=fold_dir)
        print(f"Test Metrics - Acc: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}")
        cv_results.append({'fold': fold, 'best_val_acc': best_val_acc, 'test_metrics': test_metrics})

    print("\n==== Summary of cross-validation ====")
    for metric in ['acc', 'auc', 'precision', 'recall', 'f1', 'specificity']:
        values = [result['test_metrics'][metric] for result in cv_results]
        print(f"Average {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    main()
