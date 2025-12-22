import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import random
import pandas as pd
import argparse
from utils import set_seed  # Ensure utils.py is in the same directory

# Configuration dictionary
CONFIG = {
    'model_name': 'resnet18',       # Options: cnn, densenet, senet, resnet18, resnet50, resnet152
    'modalities': ['adc', 'hbv', 't2w'], 
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_workers': 4,
    'n_splits': 5,                  # Number of cross-validation folds
    'save_features': True,          
    'feature_save_dir': 'features'
}

class MedicalVolumeDataset(Dataset):
    """
    Dataset class for loading multi-modal medical image volumes.
    Assumes a directory structure: root_dir/label/patient_id/modality/slices.
    """
    def __init__(self, root_dir, modalities, transform=None):
        self.root_dir = root_dir
        self.modalities = modalities
        self.transform = transform
        self.samples = []
        self.load_samples()

    def load_samples(self):
        for label in ['0', '1']:
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            patients = os.listdir(label_dir)
            for patient in patients:
                patient_dir = os.path.join(label_dir, patient)
                if not os.path.isdir(patient_dir):
                    continue

                patient_files = {mod: [] for mod in self.modalities}
                valid_sample = True

                for mod in self.modalities:
                    # Find modality directory (partial match)
                    modality_dirs = [d for d in os.listdir(patient_dir)
                                    if mod in d and os.path.isdir(os.path.join(patient_dir, d))]
                    
                    if not modality_dirs:
                        valid_sample = False
                        break

                    modality_dir = os.path.join(patient_dir, modality_dirs[0])
                    files = [os.path.join(modality_dir, f)
                             for f in sorted(os.listdir(modality_dir))
                             if os.path.isfile(os.path.join(modality_dir, f))]

                    if not files:
                        valid_sample = False
                        break

                    patient_files[mod] = files

                if valid_sample:
                    self.samples.append((patient_files, int(label), patient))

        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_files, label, patient_id = self.samples[idx]
        volumes = {}
        for mod in self.modalities:
            volume = self.load_and_preprocess(patient_files[mod])
            volumes[mod] = volume

        sample = {**volumes, 'label': label, 'patient_id': patient_id}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_and_preprocess(self, file_paths):
        images = []
        for file_path in file_paths:
            img = Image.open(file_path).resize((224, 224)).convert('RGB')
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            images.append(img)

        # Slice depth normalization (fixed to 25 slices)
        target_depth = 25
        if len(images) > target_depth:
            images = images[:target_depth]
        elif len(images) < target_depth:
            # Padding with the mean of existing slices
            mean_image = np.mean(images, axis=0).astype(np.float32)
            while len(images) < target_depth:
                images.append(mean_image)

        return np.stack(images).astype(np.float32)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        tensor_sample = {}
        for key in sample:
            if key == 'label':
                tensor_sample[key] = torch.tensor(sample[key]).long()
            elif key == 'patient_id':
                tensor_sample[key] = sample[key]
            else:
                # Transpose dimensions: (D, H, W, C) -> (C, D, H, W)
                # Output shape: (3, 25, 224, 224)
                tensor_sample[key] = torch.from_numpy(sample[key]).permute(3, 0, 1, 2).float()
        return tensor_sample

class PatientLevelModel(nn.Module):
    """
    Patient-level classification model incorporating a CNN backbone.
    Processes 3D volumes by processing 2D slices independently and pooling features.
    """
    def __init__(self, backbone, num_ftrs, num_classes):
        super(PatientLevelModel, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape to process slices independently: (B*D, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        
        # Extract features per slice
        slice_features = self.backbone(x)
        
        # Aggregate features: (B, D, num_ftrs) -> (B, num_ftrs) via Mean Pooling
        patient_features = slice_features.view(B, D, -1)
        patient_features_pooled = torch.mean(patient_features, dim=1)
        
        outputs = self.classifier(patient_features_pooled)
        return outputs

class FeatureExtractor(nn.Module):
    """Wrapper to extract features from the trained PatientLevelModel."""
    def __init__(self, patient_level_model):
        super(FeatureExtractor, self).__init__()
        self.backbone = patient_level_model.backbone

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        slice_features = self.backbone(x)
        patient_features = slice_features.view(B, D, -1)
        patient_features_pooled = torch.mean(patient_features, dim=1)
        return patient_features_pooled

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def get_model(model_name, num_modalities):
    """Initialize the backbone model with modified input channels."""
    in_channels = 3 * num_modalities
    num_ftrs = 0

    if model_name == 'cnn':
        model_layers = [
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]
        backbone = nn.Sequential(*model_layers)
        num_ftrs = 256
        return backbone, num_ftrs

    elif model_name == 'densenet':
        backbone = models.densenet121(pretrained=True)
        backbone.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        return backbone, num_ftrs

    elif model_name == 'senet':
        backbone = models.resnet50(pretrained=True)
        backbone.layer1 = nn.Sequential(SEBlock(256), *list(backbone.layer1.children()))
        backbone.layer2 = nn.Sequential(SEBlock(512), *list(backbone.layer2.children()))
        backbone.layer3 = nn.Sequential(SEBlock(1024), *list(backbone.layer3.children()))
        backbone.layer4 = nn.Sequential(SEBlock(2048), *list(backbone.layer4.children()))
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, num_ftrs

    elif model_name in ['resnet18', 'resnet50', 'resnet152']:
        if model_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
        else:
            backbone = models.resnet152(pretrained=True)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, num_ftrs

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def extract_features(model, feature_extractor, data_loader, device):
    feature_extractor.eval()
    features_list = []
    labels_list = []
    patient_ids_list = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = torch.cat([batch[mod] for mod in CONFIG['modalities']], dim=1).to(device)
            labels = batch['label']
            patient_ids = batch['patient_id']

            features = feature_extractor(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels.numpy())
            patient_ids_list.extend(patient_ids)

    features = np.concatenate(features_list, axis=0)
    return features, labels_list, patient_ids_list

def save_features(features, labels, patient_ids, fold_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    feature_cols = [f'Frontal_{i + 1}' for i in range(features.shape[1])]
    data = {'PTID': patient_ids, 'DX_bl': labels}
    for i, col in enumerate(feature_cols):
        data[col] = features[:, i]
    
    df = pd.DataFrame(data)
    save_path = os.path.join(save_dir, f'fold_{fold_idx}_features.csv')
    df.to_csv(save_path, index=False)
    print(f"Features saved to: {save_path}")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []
    
    # Using GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    for batch in train_loader:
        inputs = torch.cat([batch[mod] for mod in CONFIG['modalities']], dim=1).to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    # Validation Phase
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = torch.cat([batch[mod] for mod in CONFIG['modalities']], dim=1).to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    metrics = {
        'train_loss': train_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
        'train_acc': accuracy_score(train_labels, train_preds),
        'val_acc': accuracy_score(val_labels, val_preds),
        'train_auc': roc_auc_score(train_labels, train_preds),
        'val_auc': roc_auc_score(val_labels, val_preds)
    }
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='picai', help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='features', help='Directory to save extracted features')
    args = parser.parse_args()
    
    CONFIG['feature_save_dir'] = args.output_dir

    set_seed(37)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([ToTensor()])
    dataset = MedicalVolumeDataset(
        root_dir=args.data_dir, 
        modalities=CONFIG['modalities'], 
        transform=transform
    )

    n_splits = CONFIG['n_splits']
    indices = np.arange(len(dataset))
    fold_size = len(dataset) // n_splits
    
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_splits)]

    all_fold_features = []
    all_fold_labels = []
    all_fold_ids = []

    for test_fold_idx in range(n_splits):
        print(f'\n=== Processing Fold {test_fold_idx + 1} ===')
        
        val_fold_idx = (test_fold_idx + 1) % n_splits
        test_indices = folds[test_fold_idx]
        val_indices = folds[val_fold_idx]
        
        train_indices = []
        for i in range(n_splits):
            if i != test_fold_idx and i != val_fold_idx:
                train_indices.extend(folds[i])

        train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], sampler=SubsetRandomSampler(train_indices), num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], sampler=SubsetRandomSampler(val_indices), num_workers=CONFIG['num_workers'], pin_memory=True)
        test_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], sampler=SubsetRandomSampler(test_indices), num_workers=CONFIG['num_workers'], pin_memory=True)

        backbone, num_ftrs = get_model(CONFIG['model_name'], len(CONFIG['modalities']))
        model = PatientLevelModel(backbone, num_ftrs, num_classes=2).to(device)
        feature_extractor = FeatureExtractor(model).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

        best_val_auc = 0
        patience = 5
        patience_counter = 0

        for epoch in range(CONFIG['num_epochs']):
            metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
            print(f'Epoch {epoch + 1}: Train Loss: {metrics["train_loss"]:.4f}, Val AUC: {metrics["val_auc"]:.4f}')

            if metrics['val_auc'] > best_val_auc:
                best_val_auc = metrics['val_auc']
                patience_counter = 0
                torch.save({'model_state_dict': model.state_dict()}, f'best_model_fold_{test_fold_idx + 1}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break

        # Load best model for feature extraction
        checkpoint = torch.load(f'best_model_fold_{test_fold_idx + 1}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        feature_extractor = FeatureExtractor(model).to(device)

        if CONFIG['save_features']:
            print(f'Extracting features for Fold {test_fold_idx + 1} test set...')
            test_features, test_labels, test_ids = extract_features(model, feature_extractor, test_loader, device)
            all_fold_features.append(test_features)
            all_fold_labels.extend(test_labels)
            all_fold_ids.extend(test_ids)

    if CONFIG['save_features']:
        print('\nAggregating and saving strictly inductive features...')
        final_features = np.concatenate(all_fold_features, axis=0)
        save_features(final_features, all_fold_labels, all_fold_ids, 'all_folds', CONFIG['feature_save_dir'])

if __name__ == '__main__':
    main()