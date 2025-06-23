import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Union, Callable
from itertools import cycle
from tqdm import tqdm 
from detection_methods_posthoc import detection_method
from sklearn.metrics import roc_curve

class MLP(nn.Module):
    """The MLP model used for classification."""

    class Block(nn.Module):
        """The main building block of `MLP`."""
        
        def __init__(self, d_in: int, d_out: int, activation: Callable, dropout: float) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)
            self.activation = activation
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(self, d_in: int, d_layers: List[int], activation: Union['str', Callable], 
                 dropouts: Union[float, List[float]], d_out: int) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        if isinstance(activation, str):
            if activation == 'relu':
                activation = nn.ReLU()
            elif activation == 'elu':
                activation = nn.ELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        
        self.blocks = nn.Sequential(
            *[MLP.Block(
                d_in=d_layers[i - 1] if i > 0 else d_in,
                d_out=d_layers[i],
                activation=activation,
                dropout=dropouts[i]
            ) for i in range(len(d_layers))]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

        self.X_train = None
        self.X_ood = None
        self.y_train = None

    def forward(self, x: Tensor) -> Tensor:
        x_features = self.blocks(x)
        x_linear = self.head(x_features)
        return x_linear, x_features

    def get_fc(self):
        fc = self.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy(), fc
    
    def train_model(self, X: Tensor, y: Tensor, 
                    num_epochs: int = 100, lr: float = 0.005, batch_size: int = 32, test_iter: int = None) -> None:
        """Train the MLP model."""
        test_iter = test_iter if test_iter is not None else num_epochs + 1

        # store training data 
        self.X_train = X
        self.y_train = y

        # Assuming X and y are NumPy arrays or similar
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)  # Ensure correct type for classification (long for labels)
        
        # Check that X and y have the same number of samples
        assert X.size(0) == y.size(0), "Size mismatch between features (X) and labels (y)"
        
        # Create a DataLoader
        dataset = TensorDataset(X, y)  # Combine features and labels into a dataset
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 5, gamma=0.1)  

        for epoch in tqdm(range(num_epochs)):
            self.train()  # Set the model to training mode

            for batch_X, batch_y in data_loader:  # Iterate through batches
                optimizer.zero_grad()  # Clear the gradients
                outputs, _ = self(batch_X)  # Forward pass
                loss = criterion(outputs, batch_y)  # Calculate loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            lr_scheduler.step()  # Step the learning rate scheduler

            # Print the loss every few epochs
            if (epoch + 1) % test_iter == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
                
    def train_model_oe(self, X: Tensor, y: Tensor, X_ood: Tensor,
                    num_epochs: int = 100, lr: float = 0.005, batch_size: int = 32, oe_batch_size = 16, test_iter: int = None) -> None:
        """Train the MLP model with Outlier Exposure (OE)."""
        test_iter = test_iter if test_iter is not None else num_epochs + 1

                # store training data 
        self.X_train = X
        self.y_train = y
        self.X_ood = X_ood
        
        # Create a DataLoader
        dataset = TensorDataset(X, y)  # Combine features and labels into a dataset
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        dataset_out = TensorDataset(X_ood)
        data_loader_out = DataLoader(dataset_out, batch_size=oe_batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 5, gamma=0.1)  

        for epoch in tqdm(range(num_epochs)):
            self.train()  # Set the model to training mode

            for data_in, data_out in zip(data_loader, data_loader_out):  # Iterate through batches
                batch_X, batch_y = data_in[0], data_in[1]
                batch_out = data_out[0]
                
                optimizer.zero_grad()  # Clear the gradients

                # Concatenate inputs
                combined_input = torch.cat((batch_X, batch_out), dim=0)

                # Forward pass for combined input
                combined_outputs, _ = self(combined_input)  # Forward pass

                # Separate outputs
                outputs_in = combined_outputs[:batch_X.size(0)]  # In-distribution outputs
                outputs_out = combined_outputs[batch_X.size(0):]  # Out-of-distribution outputs

                # Calculate losses
                loss_in = criterion(outputs_in, batch_y)  # In-distribution loss

                # OOD loss calculation
                loss_ood = 0.5 * -(outputs_out.mean(1) - torch.logsumexp(outputs_out, dim=1)).mean()  # OOD loss

                # Combine the losses
                loss = loss_in + loss_ood

                # Backpropagation
                loss.backward()  # Backpropagation for both losses
                optimizer.step()  # Update weights

            lr_scheduler.step()  # Step the learning rate scheduler

            # Print the loss every few epochs
            if (epoch + 1) % test_iter == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    
    def predict_with_ood(self, x: Tensor, detector: str, threshold_method: str = 'percentile', percentile: float = 95, level:float = 95, **args) -> Tensor:
        """Predict labels with OOD detection. 
        Label 0 for OOD and 1-num_classes for ID predictions.
        
        Args:
            x (Tensor): Input data.
            detector (str): The name of the detection method or postprocessor.
            threshold_method (str): The method used to calculate the threshold.
            percentile (float): The percentile for threshold calculation when using the 'percentile' method.
            **args(dict): Additional arguments specific to the selected detector.
        
        Returns:
            Tensor: Tensor of predicted labels (0 for OOD, 1-num_classes for ID).
        """
        score_function = detection_method(detector=detector, model=self, **args)
        
        # Get ID confidences
        _, id_conf = score_function(self, self.X_train)
        
        # Check for OOD data and get OOD confidences
        out_conf = np.array([])  # Initialize to avoid reference error if self.X_ood is None
        if self.X_ood is not None:
            _, out_conf = score_function(self, self.X_ood)
    
        # Determine threshold
        if threshold_method == 'percentile':
            # Calculate the threshold from id_conf
            threshold = np.percentile(id_conf, percentile)
        elif threshold_method == 'roc':
            # Calculate the threshold from the ROC curve
            y_true = np.array([1] * len(id_conf) + [0] * len(out_conf))
            y_scores = np.concatenate((id_conf, out_conf))
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)

            tpr_fpr_diff = tpr - fpr
            max_diff = np.max(tpr_fpr_diff)
            
            # Find indices where the difference is equal to the maximum
            optimal_indices = np.where(tpr_fpr_diff == max_diff)[0]
            
            # If there are multiple thresholds with the same max difference, take the median of those thresholds
            if len(optimal_indices) > 1:
                threshold = np.median(thresholds[optimal_indices])
            else:
                threshold = thresholds[optimal_indices[0]]  # Use the single threshold if only one is found
        elif threshold_method =='FPR':
            y_true = np.array([1] * len(id_conf) + [0] * len(out_conf))
            y_scores = np.concatenate((id_conf, out_conf))
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
            target_fpr = (100 - level) / 100  # FPR@95 means target FPR = 5%
            valid_indices = np.where(fpr <= target_fpr)[0]
            if len(valid_indices) == 0:
                threshold = np.min(thresholds)  # Fallback to lowest threshold
            else:
                threshold = thresholds[valid_indices[-1]]  # Last index with FPR <= target
        else:
            raise ValueError(f"Unsupported threshold calculation method: {threshold_method}")
        
        # Calculate the OOD score for the input data x
        pred, conf = score_function(self, x)
    
        # Apply OOD thresholding: Assign 0 for OOD if confidence is below the threshold
        ood_mask = conf < threshold
        pred[ood_mask] = 0  # Set OOD samples to label 0
    
        # Shift class labels for ID predictions (1 to num_classes)
        pred[~ood_mask] += 1  # Shift classes to start from 1
        return pred