import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib  
import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, criterion, optimizer, device, scaler_path='scaler.pkl', epochs=20):
    model.to(device)

    # Initialize scaler
    scaler = StandardScaler()
    fitted_scaler = None

    # Track losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.numpy(), labels.numpy()

            # Fit the scaler on training features and transform
            if fitted_scaler is None:
                fitted_scaler = scaler.fit(features)
                joblib.dump(fitted_scaler, scaler_path)  # Save the fitted scaler
            
            scaled_features = torch.tensor(fitted_scaler.transform(features), dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

            # Reshape features to (batch_size, sequence_length, input_size)
            scaled_features = scaled_features.unsqueeze(1)  # Adding a sequence dimension
            
            optimizer.zero_grad()
            outputs = model(scaled_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.numpy(), labels.numpy()

                # Transform validation features using the fitted scaler
                scaled_features = torch.tensor(fitted_scaler.transform(features), dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

                # Reshape features to (batch_size, sequence_length, input_size)
                scaled_features = scaled_features.unsqueeze(1)  # Adding a sequence dimension
                
                outputs = model(scaled_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Record losses
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plot train and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


