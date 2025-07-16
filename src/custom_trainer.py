import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput


class CustomTrainer(Trainer):
    """
    Custom Trainer class extending Hugging Face's Trainer with additional functionality:
    - Support for class weights to handle imbalanced datasets
    - Custom loss computation with weighted cross-entropy
    - Threshold optimization for binary classification
    - Custom prediction with threshold application
    """

    def __init__(self, *args, class_weights=None, weights_dtype=torch.float32, **kwargs):
        """
        Initialize the CustomTrainer.

        Args:
            class_weights (array-like, optional): Weights for each class to handle class imbalance.
            weights_dtype (torch.dtype): Data type for the class weights tensor.
            *args, **kwargs: Arguments passed to the parent Trainer class.
        """
        super().__init__(*args, **kwargs)
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=weights_dtype).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss with optional class weighting.

        Args:
            model: The model to train
            inputs: The inputs and targets of the model
            return_outputs (bool): Whether to return the outputs along with the loss
            num_items_in_batch: Not used but kept for compatibility

        Returns:
            torch.Tensor or tuple: Loss value alone or with model outputs
        """
        # Extract labels
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)

        # Extract logits
        logits = outputs.get("logits")

        # Compute loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def compute_best_threshold(self, dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Find the optimal classification threshold that maximizes macro F1 score.

        Args:
            dataset: The dataset to use for threshold optimization
            ignore_keys (list, optional): Keys to ignore in the model outputs
            metric_key_prefix (str): Prefix for metric keys in the output

        Returns:
            float: The optimal threshold value
        """
        # Get raw predictions from parent class
        output = super().predict(dataset, ignore_keys, metric_key_prefix)

        # Convert logits to probabilities using softmax (for binary classification)
        logits = output.predictions
        logits_tensor = torch.tensor(logits)
        probabilities = torch.softmax(logits_tensor, dim=-1).numpy()

        # Calculate optimal threshold
        labels = output.label_ids
        thresholds = np.linspace(0.1, 0.9, 100)

        best_threshold = 0.5  # Default threshold
        best_f1 = 0

        for threshold in thresholds:
            predictions = (probabilities[:, 1] >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Return the best threshold found
        return best_threshold

    def predict(self, dataset, threshold: float = 0.5, ignore_keys=None, metric_key_prefix="test"):
        """
        Generate predictions with a custom threshold for binary classification.

        Args:
            dataset: The dataset to generate predictions for
            threshold (float): The classification threshold (default: 0.5)
            ignore_keys (list, optional): Keys to ignore in model outputs
            metric_key_prefix (str): Prefix for metric keys in the output

        Returns:
            PredictionOutput: Object containing predictions, labels, and metrics
        """
        # Get raw predictions from parent class
        output = super().predict(dataset, ignore_keys, metric_key_prefix)

        # Convert logits to probabilities using softmax (for binary classification)
        logits = output.predictions
        logits_tensor = torch.tensor(logits)
        probabilities = torch.softmax(logits_tensor, dim=-1).numpy()

        final_predictions = (probabilities[:, 1] >= threshold).astype(int)

        # Update predictions in the output object
        return PredictionOutput(predictions=final_predictions, label_ids=output.label_ids, metrics=output.metrics)
