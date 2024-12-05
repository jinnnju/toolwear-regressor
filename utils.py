import numpy as np  
import torch  


def prepare_data(loader):
    """
    PyTorch DataLoader에서 데이터를 추출하여 NumPy 배열로 변환

    Args:
        loader (torch.utils.data.DataLoader): PyTorch DataLoader 객체
    
    Returns:
        tuple: (features, labels)
        - features (np.ndarray): 입력 데이터
        - labels (np.ndarray): 타겟 데이터
    """
    features, labels = [], []
    for batch_features, batch_labels in loader:
        features.extend(batch_features.numpy())  # PyTorch Tensor → NumPy 배열
        labels.extend(batch_labels.numpy())
    return np.array(features), np.array(labels)
