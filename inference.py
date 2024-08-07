import torch
from model import TokenClassificationModel
from transformers import LayoutLMv3TokenizerFast
from utils import visualize_predictions, load_doclay, to_numpy, aggregate_outputs
from config import TrainConfig
from functools import reduce
import numpy as np
import onnxruntime
import time
from config import logger


def load_model(model_path: str):
    """
    Load the inference model from the given model_path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        TokenClassificationModel: The loaded model.

    """
    # Load the tokenizer
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        TrainConfig.base_model_name)
    weights = torch.load(model_path)
    model = TokenClassificationModel(weights['config'])
    trained_state_dict = {k.replace("_orig_mod.", ""): v for k,
                          v in weights['model'].items()}
    model.load_state_dict(trained_state_dict)
    return model, tokenizer


def run_onnx_model(data: list):
    """
    Run the ONNX model on the given input.

    Args:
        input: The input to the model.

    Returns:
        The output of the model.
    """
    # compute ONNX Runtime output prediction
    ort_inputs = {input_obj.name: to_numpy(
        input) for input_obj, input in zip(ort_session.get_inputs(), data)}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = torch.tensor(ort_outs[0])
    return ort_outs


def run_torch_model(model: TokenClassificationModel, data: dict):
    """
    Run the PyTorch model on the given input.

    Args:
        model: The model to run.
        data: The input to the model.

    Returns:
        The output of the model.
    """
    with torch.no_grad():
        outputs = model(**data, return_dict=True)
    return outputs.logits


def predict(file_path: str, config: TrainConfig, model: TokenClassificationModel, tokenizer: LayoutLMv3TokenizerFast):
    """
    Predicts the labels for the given file using the specified model and tokenizer.

    Args:
        file_path (str): The path to the input file.
        config (Config): The configuration object containing class list and other parameters.
        model: The pre-trained model for prediction.
        tokenizer: The tokenizer used for encoding the input.

    Returns:
        None
    """
    label_list = config.class_list
    label_map = {label: i for i, label in enumerate(label_list)}
    sentences, bboxes, labels, image_array = load_doclay(file_path, label_map)
    file_name = os.path.basename(file_path).split(".")[0]
    encoding = tokenizer(
        text=sentences,
        boxes=bboxes,
        padding="max_length",
        truncation=True,
        max_length=512,  # Fixed to 512 tokens
        return_tensors="pt",
        return_overflowing_tokens=True,
        stride=config.stride,  # Overlap between chunks
        return_offsets_mapping=True)
    encoding.pop("overflow_to_sample_mapping")
    encoding.pop("offset_mapping")
    encoding = {k: v.type(torch.long).to(model.device)
                if k != 'bbox' else v.to(model.device) for k, v in encoding.items()}
    offset = torch.frac(encoding["bbox"])
    encoding["bbox"] = encoding["bbox"].long()
    outputs = run_onnx_model(
        [encoding["input_ids"], encoding["bbox"], encoding["attention_mask"]])
    # outputs = run_torch_model(model, encoding)
    outputs = torch.nn.functional.softmax(outputs, dim=-1)
    out_tokens, bboxs, preds = aggregate_outputs(prob_tensor=outputs, token_ids=encoding["input_ids"], bboxes=encoding["bbox"]+offset,
                                                 chunk_size=config.max_token_len,
                                                 overlap_size=config.stride)
    prev_bbox = [-1, -1, -1, -1]
    bbox_2_token = {}
    for (token_id, bbox, pred) in zip(out_tokens, bboxs, preds):
        bbox = bbox.tolist()
        token_id = token_id.item()
        if bbox == tokenizer.pad_token_box:
            continue
        if bbox != prev_bbox:
            bbox_2_token[tuple(bbox)] = []
            bbox_2_token[tuple(bbox)].append((token_id, pred))
            prev_bbox = bbox
        else:
            bbox_2_token[tuple(bbox)].append((token_id, pred))

    final_pred_list = []
    for bbox, out in bbox_2_token.items():
        token_list, pred_list = [], []
        for token_id, pred in out:
            token_list.append(tokenizer.decode(token_id))
            pred_list.append(pred)
        Text = "".join(token_list)
        Line_pred = np.argmax([reduce(lambda x, y: x * y, prob)
                               for prob in np.transpose(pred_list)])
        final_pred_list.append((bbox, label_list[Line_pred]))
        logger.info(f"Text: {Text}, Prediction: {label_list[Line_pred]}")
    visualize_predictions(image_array, final_pred_list, file_name)


if __name__ == "__main__":
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ort_session = onnxruntime.InferenceSession(
        "weights/quantized_best.onnx", providers=["CPUExecutionProvider"])
    model_path = "weights/best.pt"
    model, tokenizer = load_model(model_path)
    model.to(device)
    model.eval()
    folder_path = "dataset/test/annotations"
    file_list = os.listdir(folder_path)
    start_time = time.time()
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        predict(file_path, TrainConfig, model, tokenizer)
    logger.info(f"Avg Time taken: {(time.time() - start_time)/len(file_list)}")
