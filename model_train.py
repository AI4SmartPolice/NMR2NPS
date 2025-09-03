import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rdkit import Chem
from rdkit import RDLogger
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Load datasets
excel_file = 'SFT_data.xlsx'
smiles_file = 'CL_train_data.xlsx'
data_model = pd.read_excel(excel_file, sheet_name='model', header=0)
data_test = pd.read_excel(excel_file, sheet_name='test', header=0)
data_smiles = pd.read_excel(smiles_file, sheet_name='model', header=0)

# Function to filter columns and replace low values
def filter_and_replace_data(df):
    df = df.copy()
    numeric_cols = [col for col in df.columns if isinstance(col, (int, float)) and col <= 8.5]
    selected_cols = [col for col in df.columns if col in numeric_cols or not isinstance(col, (int, float))]
    df = df[selected_cols]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].where(df[numeric_cols] >= 0.0005, 0.0005)
    return df

# Process SMILES data
def process_smiles(df):
    df = df.copy()
    df['SMILES'] = df['SMILES'].str.strip()
    valid_smiles = []
    valid_indices = []
    for idx, smile in enumerate(df['SMILES']):
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                valid_smiles.append(smile)
                valid_indices.append(idx)
        except:
            continue
    df = df.iloc[valid_indices].reset_index(drop=True)
    return df

# Apply filtering and SMILES processing
data_model = filter_and_replace_data(data_model)
data_test = filter_and_replace_data(data_test)
data_smiles = process_smiles(data_smiles)

# Preprocessing functions
def normalization(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def fourth_root(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.abs(X)
    return np.power(X + 1e-6, 0.25)

def norm_plus_fourth_root(X):
    X_norm = normalization(X)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return np.power(X_norm + 1e-6, 0.25)

# Preprocess NMR data
def preprocess_data(df, method):
    df = df.copy()
    df.columns = [str(col) for col in df.columns]
    le = LabelEncoder()
    y = le.fit_transform(df['Class'].values)
    id_sample = df[['id', 'sample']].copy() if 'id' in df.columns and 'sample' in df.columns else None
    df = df.loc[:, ~df.columns.str.contains('id', na=False)]
    X_df = df.drop(['Class', 'sample'], axis=1, errors='ignore')
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    X_numeric = X_df[numeric_cols].values
    X_numeric = np.where(np.isnan(X_numeric), np.nanmean(X_numeric, axis=0), X_numeric)
    
    if method == 'N':
        X_processed = normalization(X_numeric)
    elif method == '4R':
        X_processed = fourth_root(X_numeric)
    elif method == 'N+4R':
        X_processed = norm_plus_fourth_root(X_numeric)
    
    X_processed = X_processed.reshape(X_processed.shape[0], 1, X_processed.shape[1])
    return X_processed, y, le, id_sample

# Load ChemBERTa model
chemberta_model = AutoModel.from_pretrained("chemberta")
chemberta_tokenizer = AutoTokenizer.from_pretrained("chemberta")

# Preprocess SMILES for ChemBERTa
def preprocess_smiles_for_chemberta(smiles_list, max_length=128):
    encodings = chemberta_tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.size()
        x_pool = self.avg_pool(x).view(b, c)
        x = self.relu(self.fc1(x_pool))
        x = self.sigmoid(self.fc2(x))
        x = x.view(b, c, 1)
        return x

    # Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, normalize_features=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.normalize_features = normalize_features

    def forward(self, output1, output2, label):

        if self.normalize_features:
            output1 = F.normalize(output1, p=2, dim=1)
            output2 = F.normalize(output2, p=2, dim=1)

        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_same = label * (euclidean_distance ** 2)
        loss_diff = (1 - label) * (torch.clamp(self.margin - euclidean_distance, min=0.0) ** 2)
        
        return torch.mean(loss_same + loss_diff)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(0.3)
        self.attention = ChannelAttention(out_channels)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        attention_weights = self.attention(out)
        out = out * attention_weights
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out

# CNN Model for NMR data
class CNNModel(nn.Module):
    def __init__(self, input_channels, timesteps, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2, stride=2, padding=0)
        self.res1 = ResidualBlock(16, 16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2, stride=2, padding=0)
        self.res2 = ResidualBlock(32, 32)
        out_size = timesteps // 4
        self.fc1 = nn.Linear(32 * out_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.res1(x)
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.res2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_features(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.res1(x)
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.res2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def train_evaluate_model(X_train_nmr, y_train, X_train_smiles, y_train_smiles, X_test_nmr, y_test, dataset_name, method, label_mapping, id_sample_test, seed):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y_train))
    timesteps = X_train_nmr.shape[2]
    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    misclassified_data = []
    
    # Convert to tensors
    X_train_nmr_tensor = torch.FloatTensor(X_train_nmr).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_nmr_tensor = torch.FloatTensor(X_test_nmr).to(device)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Randomly select NMR subset to match SMILES length
    num_smiles_samples = len(y_train_smiles)
    indices = np.random.choice(len(y_train), size=num_smiles_samples, replace=False)
    X_train_nmr_subset = X_train_nmr[indices]
    y_train_subset = y_train[indices]
    X_train_nmr_subset_tensor = torch.FloatTensor(X_train_nmr_subset).to(device)
    y_train_subset_tensor = torch.LongTensor(y_train_subset).to(device)
    y_train_smiles_tensor = torch.LongTensor(y_train_smiles).to(device)
    
    # Initialize models
    cnn_model = CNNModel(input_channels=1, timesteps=timesteps, num_classes=num_classes).to(device)
    chemberta = chemberta_model.to(device)
    chemberta_fc = nn.Linear(chemberta.config.hidden_size, 128).to(device)
    
    # Contrastive pretraining
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(list(cnn_model.parameters()) + list(chemberta.parameters()) + list(chemberta_fc.parameters()), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    max_contrastive_epochs = 50
    
    cnn_model.train()
    chemberta.train()
    chemberta_fc.train()
    
    for epoch in range(max_contrastive_epochs):
        optimizer.zero_grad()
        
        nmr_features = cnn_model.get_features(X_train_nmr_subset_tensor)
        smiles_features = chemberta_fc(chemberta(**X_train_smiles).pooler_output)
        
        labels = (y_train_subset_tensor == y_train_smiles_tensor).float()
        cont_loss = contrastive_criterion(nmr_features, smiles_features, labels)
        
        cont_loss.backward()
        optimizer.step()
        scheduler.step(cont_loss)
        
        if cont_loss.item() < best_loss:
            best_loss = cont_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Contrastive pretraining stopped early at epoch {epoch+1}")
                break
    print(f"cl_best_loss:{best_loss}")
    
    torch.save(cnn_model.state_dict(), f'pro_cnn_predictions/pretrain_model_weights_{method}_seed_{seed}.pth')
    # Finetune CNN with classification loss
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)
    scheduler_cnn = ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.1, patience=3, verbose=True)
    
    cnn_model.train()
    warmup_epochs = 5
    for epoch in range(150):
        if epoch < warmup_epochs:
            lr = 0.001 * (epoch + 1) / warmup_epochs
            for param_group in optimizer_cnn.param_groups:
                param_group['lr'] = lr
        optimizer_cnn.zero_grad()
        
        outputs = cnn_model(X_train_nmr_tensor)
        cls_loss = criterion(outputs, y_train_tensor)
        
        cls_loss.backward()
        optimizer_cnn.step()
        scheduler_cnn.step(cls_loss)
    
    # Save CNN model weights
    torch.save(cnn_model.state_dict(), f'pro_cnn_predictions/model_weights_{method}_seed_{seed}.pth')
    
    # Evaluation
    cnn_model.eval()
    with torch.no_grad():
        y_pred = cnn_model(X_test_nmr_tensor).argmax(dim=1).cpu().numpy()
    
    # Convert predictions to text labels
    y_pred_labels = [label_mapping.get(y, str(y)) for y in y_pred]
    y_test_labels = [label_mapping.get(y, str(y)) for y in y_test]
    
    # Create DataFrame for predictions
    if id_sample_test is not None:
        pred_df = pd.DataFrame({
            'id': id_sample_test['id'],
            'sample': id_sample_test['sample'],
            'Class': y_test_labels,
            'y_pred': y_pred_labels
        })
    else:
        pred_df = pd.DataFrame({
            'id': range(len(y_test)),
            'sample': ['unknown'] * len(y_test),
            'Class': y_test_labels,
            'y_pred': y_pred_labels
        })
    
    # Save predictions to CSV
    pred_df.to_csv(f'pro_cnn_predictions/predictions_{method}_seed_{seed}.csv', index=False)
    
    # Record misclassified samples
    misclassified_indices = np.where(y_test != y_pred)[0]
    for idx in misclassified_indices:
        misclassified_data.append({
            'Dataset': dataset_name,
            'Method': method,
            'Seed': seed,
            'Sample_Index': idx,
            'True_Label': y_test[idx],
            'Predicted_Label': y_pred[idx]
        })
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    
    print(f"\n=== Results for {dataset_name} with {method} (Seed {seed}) ===")
    for metric in results:
        mean = np.mean(results[metric])
        print(f"{metric.capitalize()}: {mean:.4f}")
    
    return results, misclassified_data

if __name__ == "__main__":
    methods = ['4R', 'N+4R']
    seeds = [42, 123, 456]
    all_results = {}
    all_misclassified = []
    
    # Generate label_mapping
    label_mapping = {i: label for i, label in enumerate(pd.concat([data_model['Class'], data_test['Class']]).unique())}
    
    # Preprocess NMR data (full dataset)
    X_train, y_train, le, _ = preprocess_data(data_model, 'N')
    X_test, y_test, _, id_sample_test = preprocess_data(data_test, 'N')
    
    # Preprocess SMILES data (only for training)
    smiles_train = preprocess_smiles_for_chemberta(data_smiles['SMILES'].tolist())
    y_train_smiles = le.transform(data_smiles['Class'].values)
    
    # Move SMILES tensors to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles_train = {k: v.to(device) for k, v in smiles_train.items()}
    
    # Preprocess NMR data and train
    for method in methods:
        X_train, y_train, _, _ = preprocess_data(data_model, method)
        X_test, y_test, _, id_sample_test = preprocess_data(data_test, method)
        
        assert X_train.shape[2] == X_test.shape[2], "Training and test datasets have different numbers of features!"
        
        method_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        method_misclassified = []
        
        for seed in seeds:
            results, misclassified = train_evaluate_model(
                X_train, y_train, smiles_train, y_train_smiles,
                X_test, y_test, 
                'Model-Test', method, label_mapping, id_sample_test, seed
            )
            for metric in results:
                method_results[metric].extend(results[metric])
            method_misclassified.extend(misclassified)
        
        # Calculate mean and std for each metric
        mean_std_results = {}
        for metric in method_results:
            mean_std_results[metric] = (np.mean(method_results[metric]), np.std(method_results[metric]))
        
        all_results[f"Model-Test - {method}"] = {
            'accuracy': mean_std_results['accuracy'][0],
            'accuracy_std': mean_std_results['accuracy'][1],
            'precision': mean_std_results['precision'][0],
            'precision_std': mean_std_results['precision'][1],
            'recall': mean_std_results['recall'][0],
            'recall_std': mean_std_results['recall'][1],
            'f1': mean_std_results['f1'][0],
            'f1_std': mean_std_results['f1'][1]
        }
        all_misclassified.extend(method_misclassified)
    
    print("\n=== Model Comparison ===")
    comparison_df = pd.DataFrame(all_results).T
    print(comparison_df)
