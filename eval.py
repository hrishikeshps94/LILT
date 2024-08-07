import evaluate
import torch
from config import TrainConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_labels(predictions, references, labels):
    """
    Get the true predictions and true labels from the given predictions and references.

    Args:
        predictions (torch.Tensor): The predictions tensor.
        references (torch.Tensor): The references tensor.
        labels (List[str]): The list of labels.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: A tuple containing the true predictions and true labels.
    """
    # Transform predictions and references tensors to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [p for (p, l) in zip(y_pred, y_true) if l != -100]
    true_labels = [l for (p, l) in zip(y_pred, y_true) if l != -100]
    return true_predictions, true_labels


# Metrics
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")


def calculate_metrics(predictions, references, labels):
    """
    Calculate precision, recall, F1 score, and accuracy metrics for a given set of predictions.

    Args:
        predictions (list): List of predicted values.
        references (list): List of reference values.
        labels (list): List of possible labels.

    Returns:
        dict: A dictionary containing the calculated metrics:
            - overall_precision: Precision score.
            - overall_recall: Recall score.
            - overall_f1: F1 score.
            - overall_accuracy: Accuracy score.
    """
    results = {}
    label_map = {i: label for i, label in enumerate(labels)}
    main_labels = list(label_map.keys())
    true_predictions, true_labels = get_labels(predictions, references, labels)

    # Calculate precision, recall, and F1
    precision = precision_metric.compute(
        predictions=true_predictions, references=true_labels, average=None, labels=main_labels, zero_division=0)['precision']
    recall = recall_metric.compute(
        predictions=true_predictions, references=true_labels, average=None, labels=main_labels, zero_division=0)["recall"]
    f1 = f1_metric.compute(predictions=true_predictions,
                           references=true_labels, average=None, labels=main_labels)["f1"]
    accuracy = accuracy_metric.compute(
        predictions=true_predictions, references=true_labels)["accuracy"]
    required_label_ids = [
        i for i in label_map.keys() if i not in TrainConfig.remove_labels]
    results["overall_accuracy"] = round(accuracy, 4)
    results["overall_precision"] = precision[required_label_ids].mean().round(4)
    results["overall_recall"] = recall[required_label_ids].mean().round(4)
    results["overall_f1"] = f1[required_label_ids].mean().round(4)
    for label_id, label in enumerate(main_labels):
        if label_id in TrainConfig.remove_labels:
            continue
        results[label_map[label_id]] = {
            "precision": precision[label_id].round(4),
            "recall": recall[label_id].round(4),
            "f1": f1[label_id].round(4)
        }
    return results
