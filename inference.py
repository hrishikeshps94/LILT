from utils import visualize_predictions, aggregate_outputs
import onnxruntime
import os
from transformers import LayoutLMv3TokenizerFast
from local_infer import predict_page
from config import logger
import json
import torch
from model import TokenClassificationModel


def model_fn(model_dir):
    """
    Load the model and tokenizer for inference.

    Args:
        model_dir (str): The directory path where the model is stored.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    # Load the model
    load_path = os.path.join(model_dir, "quantized_best.onnx")
    model = onnxruntime.InferenceSession(
        load_path, providers=["CPUExecutionProvider"])
    # logger.info(f"Loading model from {model_dir}")
    # load_path = os.path.join(model_dir, "best.pt")
    # weights = torch.load(load_path, map_location="cpu")
    # model = TokenClassificationModel(weights['config'])
    # trained_state_dict = {k.replace("_orig_mod.", ""): v for k,
    #                       v in weights['model'].items()}
    # model.load_state_dict(trained_state_dict)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        os.path.join(model_dir, "tokenizer"))
    return model, tokenizer


def input_fn(request_body, request_content_type):
    logger.info(
        f"Request content type: {request_content_type}, received request body: {request_body}")
    if request_content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError(
            'Content type must be application/json. Provided: {0}'.format(request_content_type))


def predict_fn(data, model):
    """
    Make predictions on the input data.

    Args:
        data (dict): The input data. List of dictionaries containing the text and bbox.
        model (tuple): The loaded model and tokenizer.

    Returns:
        dict: The predictions.
    """

    logger.info(f"Processing started")
    # Predict for each page
    # Model , data, tokenizer
    predictions = predict_page(model[0], data, model[1])
    logger.info(f"Processing Completed , Predictions: {predictions}")

    return predictions


def output_fn(prediction, accept):
    logger.info(f"Output data of type {accept} returned: {prediction}")
    if accept == 'application/json':
        return json.dumps({'generated_text': prediction})
    else:
        raise ValueError(
            "Unsupported content type: {}".format(accept))
