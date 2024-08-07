from config import logger
import math
import cv2
import os
import torch
import json


def get_lr(it, config, iterations_per_epoch):
    """
    Calculate the learning rate based on the current iteration.

    Args:
        it (int): The current iteration.
        config (object): The configuration object containing the learning rate parameters.
        iterations_per_epoch (int): The number of iterations per epoch.

    Returns:
        float: The calculated learning rate.

    Raises:
        AssertionError: If the decay ratio is not within the range [0, 1].

    """
    # Linear warmup for intial warmup steps
    warmup_steps = config.warmup_epochs * iterations_per_epoch
    max_steps = config.epochs * iterations_per_epoch
    if it < warmup_steps:
        return config.max_lr * (it+1)/warmup_steps
    if it > max_steps:
        return config.max_lr * 0.1

    decay_ratio = (it - warmup_steps) / \
        (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"Decay ratio must be in [0,1] but got {decay_ratio}"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.max_lr*0.1 + coeff * (config.max_lr - config.max_lr*0.1)


def normalize_bbox(bbox, width, height):
    """
    Normalizes the bounding box coordinates.

    Args:
        bbox (list): The bounding box coordinates.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        list: The normalized bounding box coordinates.
    """
    assert len(bbox) == 4, f"The bounding box is invalid: {bbox}"
    assert bbox[0] < bbox[2], f"The bounding box is invalid: {bbox}"
    assert bbox[1] < bbox[3], f"The bounding box is invalid: {bbox}"
    return [
        bbox[0] / width * 1000,
        bbox[1] / height * 1000,
        bbox[2] / width * 1000,
        bbox[3] / height * 1000
    ]


def denormalize_bbox(norm_bbox, width, height):
    """
    Denormalizes the bounding box coordinates.

    Args:
        norm_bbox (list): The normalized bounding box coordinates.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        list: The denormalized bounding box coordinates.
    """
    assert len(norm_bbox) == 4, f"The bounding box is invalid: {norm_bbox}"
    return [
        norm_bbox[0] / 1000 * width,
        norm_bbox[1] / 1000 * height,
        norm_bbox[2] / 1000 * width,
        norm_bbox[3] / 1000 * height
    ]


def visualize_predictions(image_array, predictions, file_name):
    """
    Visualizes the predictions on the given image array.

    Args:
        image_array (numpy.ndarray): The input image array.
        predictions (list): A list of tuples containing the bounding box coordinates and labels.

    Returns:
        None
    """
    image = image_array.copy()
    # Draw each bounding box and label
    for count, (bbox, label) in enumerate(predictions):
        # Denormalize the bounding box
        bbox = denormalize_bbox(bbox, image.shape[1], image.shape[0])

        # Draw the bounding box
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (35, 168, 255), 1)
        # Draw the label
        cv2.putText(image, f"{label}", (int(bbox[2]), int(
            bbox[3] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (111, 148, 108), 1)

    # Display the image
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(os.path.join("output", f'{file_name}.jpg'), image)


def xywh2xyxy(bbox):
    """
    Converts bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (list): The bounding box coordinates in (x, y, w, h) format.

    Returns:
        list: The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def load_doclay(file_path, label_map):
    """
    Load the document layout from a JSON file and return the sentences, bounding boxes, labels, and image array.

    Args:
        file_path (str): The path to the JSON file containing the document layout.
        label_map (dict): A dictionary mapping category names to label values.

    Returns:
        tuple: A tuple containing the sentences (list), bounding boxes (list), labels (list), and image array (numpy.ndarray).
    """
    image_path = file_path.replace(
        "annotations", "images").replace("json", "png")
    image_array = cv2.imread(image_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    words, bboxes, labels = [], [], []
    for item in data["form"]:
        word_list = item["text"].split(' ')
        words.extend(word_list)
        labels.extend([label_map[item["category"]]]*len(word_list))
        bboxes.extend([normalize_bbox(
            xywh2xyxy(item["box_line"]), 1025, 1025)]*len(word_list))
    return words, bboxes, labels, image_array


def aggregate_outputs(prob_tensor, token_ids, bboxes, chunk_size=512, overlap_size=128):
    """
    Aggregates the output probabilities, token IDs, and bounding boxes from multiple chunks of data.

    Args:
        prob_tensor (torch.Tensor): The tensor containing the output probabilities for each chunk.
        token_ids (torch.Tensor): The tensor containing the token IDs for each chunk.
        bboxes (torch.Tensor): The tensor containing the bounding boxes for each chunk.
        chunk_size (int, optional): The size of each chunk. Defaults to 512.
        overlap_size (int, optional): The size of the overlap between adjacent chunks. Defaults to 128.

    Returns:
        torch.Tensor: The aggregated token IDs.
        torch.Tensor: The aggregated bounding boxes.
        torch.Tensor: The aggregated probabilities.
    """
    # Initialize the aggregated probabilities tensor
    aggr_prob_size = chunk_size + \
        (prob_tensor.size(0)-1)*(chunk_size - overlap_size)
    aggr_probs = torch.zeros(
        aggr_prob_size, prob_tensor.size(-1)).to(prob_tensor.device)
    aggr_token_ids = torch.zeros(
        aggr_prob_size, dtype=torch.long).to(token_ids.device)
    aggr_bboxes = torch.zeros(aggr_prob_size, 4).to(bboxes.device)
    # Aggregate the probabilities
    for i, (prob, token, bbox) in enumerate(zip(prob_tensor, token_ids, bboxes)):
        start = i * (chunk_size - overlap_size)
        end = start + chunk_size
        if i == 0:
            aggr_probs[start:end, ...] = prob
            aggr_token_ids[start:end] = token
            aggr_bboxes[start:end, ...] = bbox
        else:
            aggr_probs[start:start+overlap_size, ...] = (
                aggr_probs[start:start+overlap_size, ...] + prob[:overlap_size, ...]) / 2
            aggr_probs[start+overlap_size:end, ...] = prob[overlap_size:, ...]
            aggr_token_ids[start +
                           overlap_size:end] = token[overlap_size:, ...]
            aggr_bboxes[start+overlap_size:end, ...] = bbox[overlap_size:, ...]
    return aggr_token_ids.detach().cpu(), aggr_bboxes.detach().cpu(), aggr_probs.detach().cpu()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def copy_model_state(model, hugg_model):
    sd = model.state_dict()
    sd_hugg = hugg_model.state_dict()
    sd_k = model.state_dict().keys()
    sd_hugg_k = hugg_model.state_dict().keys()
    for key in sd_k:
        if key in sd_hugg_k:
            with torch.no_grad():
                sd[key].copy_(sd_hugg[key]) if sd[key].shape == sd_hugg[key].shape else logger.info(
                    f"Shape mismatch for key {key}")
        else:
            print(f"Key {key} not found in hugg_model")


if __name__ == "__main__":
    pass
