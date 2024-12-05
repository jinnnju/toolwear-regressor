import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class FeatureExtractor:
    def calculate_features(self, data):
        Z1 = np.mean(np.abs(data))  # (Mean Value)
        Z2 = np.sqrt(np.mean(data ** 2))  # RMS
        Z3 = np.std(data)  # Standard Deviation
        Z4 = Z2 / Z1  # Shape Factor
        Z5 = np.mean(((np.abs(data - Z1)) / Z3) ** 3)  # Skewness
        Z6 = np.mean(((np.abs(data - Z1)) / Z3) ** 4)  # Kurtosis
        Z7 = np.max(np.abs(data))  # Peak Value
        Z8 = Z7 / Z2  # Crest Factor
        Z9 = Z7 / Z1  # Impulse Factor
        Z10 = np.sum([(f ** 2) * p for f, p in enumerate(np.abs(np.fft.fft(data)) ** 2)])  # (MSF)
        Z11 = np.mean(np.abs(np.fft.fft(data)) ** 2)  # (MPS)
        Z12 = np.sum([f * p for f, p in enumerate(np.abs(np.fft.fft(data)) ** 2)]) / np.sum(np.abs(np.fft.fft(data)) ** 2)  # (FC)
        return [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12]

class Makeloader(Dataset, FeatureExtractor):
    def __init__(self, directories, mode):
        super().__init__()
        self.directories = directories
        self.mode = mode
        self.data, self.labels = self._load_data()

    def _load_data(self):
        all_data = []
        all_labels = []
        
        for directory in self.directories:
            acc_data = []
            force_data = []

            # Accelerometer 데이터 처리
            if self.mode in ['Acc', 'Mix']:
                acc_folder = os.path.join(directory, "Accelerometer")
                if os.path.exists(acc_folder):
                    for file in os.listdir(acc_folder):
                        if file.endswith('.csv'):
                            filepath = os.path.join(acc_folder, file)
                            csv_data = pd.read_csv(filepath)
                            instance_features = []
                            for col in csv_data.columns:
                                if csv_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                                    instance_features.extend(self.calculate_features(csv_data[col].values))
                            acc_data.append(instance_features)

            # Force 데이터 처리
            if self.mode in ['Force', 'Mix']:
                force_folder = os.path.join(directory, "Force")
                if os.path.exists(force_folder):
                    for file in os.listdir(force_folder):
                        if file.endswith('.csv'):
                            filepath = os.path.join(force_folder, file)
                            csv_data = pd.read_csv(filepath)
                            instance_features = []
                            for col in csv_data.columns:
                                if csv_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                                    instance_features.extend(self.calculate_features(csv_data[col].values))
                            force_data.append(instance_features)

            # Mix 모드 처리: Accelerometer와 Force 데이터를 concat
            if self.mode == 'Mix':
                for acc_instance, force_instance in zip(acc_data, force_data):
                    all_data.append(acc_instance + force_instance)
            else:
                # Acc 또는 Force 단일 모드 처리
                all_data.extend(acc_data if self.mode == 'Acc' else force_data)

            # Label 데이터 로드
            label_file = os.path.join(directory, f"{os.path.basename(directory.strip('/'))}_all_labels.csv")
            if os.path.exists(label_file):
                label_df = pd.read_csv(label_file)
                label_column = [col for col in label_df.columns if 'Tool Wear' in col][0]  # 'Tool Wear'가 포함된 컬럼명 찾기
                labels = label_df[label_column].values
                all_labels.extend(labels)

        # 데이터와 라벨의 개수가 같은지 확인
        if len(all_data) != len(all_labels):
            raise ValueError("Data and labels length mismatch. Check your directories and label files.")

        return pd.DataFrame(all_data), np.array(all_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].values, self.labels[idx]


def create_loaders(directories, mode, batch_size, train_val_ratio=None, shuffle=True):
    """
    Creates train, test, and optionally validation DataLoaders.

    :param directories: List of directories containing data
    :param mode: Data mode ('Acc', 'Force', or 'Mix')
    :param batch_size: Batch size for DataLoaders
    :param train_val_ratio: Ratio for splitting train into train/val (e.g., 0.9 for 90% train, 10% val)
    :param shuffle: Whether to shuffle the DataLoader
    :return: Train DataLoader, Test DataLoader, (optional Validation DataLoader)
    """
    dataset = Makeloader(directories, mode)

    if train_val_ratio:
        # Split dataset into train and validation
        train_size = int(len(dataset) * train_val_ratio)
        val_size = len(dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    else:
        # Create only train_loader
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader
