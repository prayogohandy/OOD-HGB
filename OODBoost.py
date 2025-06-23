import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, clone

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier

class OODBaseModel(BaseEstimator):
    def __init__(self, model_instance, use_class_weight: bool = False, hierarchical_model: bool = True):
        """
        Initialize the OOD Detection and ID Classification model.

        Parameters:
        - model_instance: The model class to be used (e.g., XGBClassifier, LGBMClassifier).
        - use_class_weight (bool): Whether to use class weights for imbalance handling (default: False).
        - hierarchical_model (bool): True for two-stage classification (OOD + ID), False for direct OOD+ID classification (default: True).
        """ 
        self.use_class_weight = use_class_weight
        self.hierarchical_model = hierarchical_model
    
        if self.hierarchical_model:
            self.model_ood = model_instance
            self.model_id = clone(model_instance) # independent model
        else:
            self.model = model_instance

    def calculate_class_weight(self, y):
        """Calculate class weights for both binary (OOD detection) and multi-class classification (ID classes)."""
        counter = Counter(y)
        majority_class = max(counter.values())
        return {cls: majority_class / count for cls, count in counter.items()}

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_ood: np.ndarray, **args):
        """
        Train the model based on the model type.
    
        Parameters:
        - X_train (np.ndarray): Training data for in-distribution (ID).
        - y_train (np.ndarray): Labels for in-distribution training data.
        - X_ood (np.ndarray): Out-of-distribution (OOD) training data.
        """
        if self.hierarchical_model:
            self.train_hierarchical(X_train, y_train, X_ood, **args)
        else:
            self.train_direct(X_train, y_train, X_ood, **args)
    
    def train_hierarchical(self, X_train: np.ndarray, y_train: np.ndarray, X_ood: np.ndarray, **args):
        """Train the hierarchical classifier (OOD + ID)."""
        # Combine ID and OOD data
        X_train_combined = np.vstack([X_train, X_ood])
        y_train_combined = np.hstack([y_train, -1 * np.ones(len(X_ood))])  # OOD labeled as -1
        y_train_ood = (y_train_combined != -1).astype(int)  # 1 if ID, 0 if OOD
    
        # Apply class weighting for OOD detection
        if self.use_class_weight:
            class_weights_ood = self.calculate_class_weight(y_train_ood)
            weight_pos_ood = class_weights_ood[1] / class_weights_ood[0]
        else:
            weight_pos_ood = 1
    
        # Train the OOD detection model
        self.model_ood.set_params(scale_pos_weight=weight_pos_ood)
        self.model_ood.fit(X_train_combined, y_train_ood, **args)
    
        # Classify ID classes
        if self.use_class_weight:
            class_weights_id = self.calculate_class_weight(y_train)
            sample_weights_id = np.array([class_weights_id[label] for label in y_train])
        else:
            sample_weights_id = None
    
        # Train the ID classification model
        self.model_id.fit(X_train, y_train, sample_weight=sample_weights_id, **args)
    
    def train_direct(self, X_train: np.ndarray, y_train: np.ndarray, X_ood: np.ndarray, **args):
        """Train the direct OOD + ID classifier."""
        # Combine ID and OOD data
        X_combined = np.vstack([X_train, X_ood])
        y_combined = np.hstack([y_train, -1 * np.ones(len(X_ood))])  # OOD labeled as -1
        y_combined += 1  # OOD label 0, ID starts at 1
    
        # Apply class weighting for OOD + ID classification
        if self.use_class_weight:
            class_weights_id = self.calculate_class_weight(y_combined)
            sample_weights_id = np.array([class_weights_id[label] for label in y_combined])
        else:
            sample_weights_id = None
    
        # Train the combined OOD + ID model
        self.model.fit(X_combined, y_combined, sample_weight=sample_weights_id, **args)

    def predict(self, X: np.array):
        """Predict OOD vs ID and ID classes on the provided test set."""
        if self.hierarchical_model:
            return self.predict_hierarchical(X)
        else:
            return self.predict_direct(X)

    def predict_hierarchical(self, X: np.array):
        """Make predictions using the stacked classifier (OOD + ID)."""
        # Predict OOD vs ID
        y_pred = self.model_ood.predict(X).flatten() # 0 OOD, 1 ID

        # Predict ID classes on the samples classified as ID
        X_test_id = X[y_pred == 1]
        if len(X_test_id) > 0:
            y_pred_id = self.model_id.predict(X_test_id).flatten() + 1 # ID 1-num_class
            y_pred[y_pred == 1] = y_pred_id
        return y_pred

    def predict_direct(self, X: np.array):
        """Make predictions using the combined OOD + ID classifier."""
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.array):
        """Get predicted probabilities for OOD vs ID and ID classes."""
        if self.hierarchical_model:
            return self.predict_proba_hierarchical(X)
        else:
            return self.predict_proba_direct(X)

    def predict_proba_hierarchical(self, X: np.array):
        """Get predicted probabilities for the hierarchical model."""
        prob_ood_id = self.model_ood.predict_proba(X)  # Shape: (n_samples, 2)
        prob_id = self.model_id.predict_proba(X)  # Shape: (n_samples_id, n_classes_id)

        prob_combined = np.zeros((len(X), prob_id.shape[1] + 1))  # +1 for OOD
        prob_combined[:, 0] = prob_ood_id[:, 0]  # OOD probability
        prob_combined[:, 1:] = prob_id * prob_ood_id[:, 1][:, np.newaxis]  # Multiply by ID probability
        return prob_combined

    def predict_proba_direct(self, X: np.array):
        """Get predicted probabilities using the combined OOD + ID classifier."""
        return self.model.predict_proba(X)  # Shape: (n_samples, n_classes)

class OODModel(OODBaseModel):
    """
    A class for handling Out-of-Distribution (OOD) and In-Distribution (ID) classification tasks using different model types.
    This model extends the `OODBaseModel` to define and manage the model used for OOD detection and ID classification.

    Parameters
    ----------
    model_type : str
        The type of model to use for OOD detection and ID classification. Supported values are 'xgboost', 'lightgbm', 'adaboost', and 'catboost'.
    model_instance : callable, optional
        A callable that represents the model instance to be used. If not provided, it defaults to the model specified by `model_type`.
    model_params : dict, optional
        A dictionary of parameters to be passed to the model constructor. Ignored if `model_instance` is provided.
    use_class_weight : bool, optional
        Whether to use class weights for handling imbalanced datasets (default is False).
    hierarchical_model : bool, optional
        Whether to use a hierarchical approach with separate models for OOD and ID detection (default is True). If set to False, a combined model is used.
    """
    def __init__(self, model_type: str, model_instance: callable = None,
                 model_params: dict = None, use_class_weight: bool = False,
                 hierarchical_model: bool = True):
        # Determine the model instance if not provided
        if model_instance is None:
            model_instance = self.get_model(model_type, model_params or {})

        # Call the base class constructor
        super().__init__(model_instance=model_instance,
                         use_class_weight=use_class_weight,
                         hierarchical_model=hierarchical_model)

    def get_model(self, model_type: str, model_params: dict):
        if model_type == 'xgboost':
            return XGBClassifier(random_state=0, **model_params)
        elif model_type == 'lightgbm':
            return LGBMClassifier(random_state=0, verbose=-1, **model_params)
        elif model_type == 'adaboost':
            return AdaBoostClassifier(random_state=0, **model_params)
        elif model_type == 'catboost':
            return CatBoostClassifier(random_state=0, verbose=0, **model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")