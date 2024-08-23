from utils import visualize_predictions, convert_line2word_bbox, aggregate_outputs
import onnxruntime
import os
from transformers import LayoutLMv3TokenizerFast
from config import logger
from config import TrainConfig as config
import numpy as np
from functools import reduce
from _inference import run_onnx_model, run_torch_model
import torch


def model_load(model_dir):
    """
    Load the model and tokenizer for inference.

    Args:
        model_dir (str): The directory path where the model is stored.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    # Load the model
    load_path = os.path.join(model_dir, "weights/quantized_best.onnx")
    model = onnxruntime.InferenceSession(
        load_path, providers=["CPUExecutionProvider"])
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained('weights/tokenizer')
    return model, tokenizer


def predict_page(inference_model, data, tokenizer):
    """
    Predicts the text and corresponding labels for a given page of an image.

    Args:
        inference_model (object): The inference model used for prediction.
        data (list): The data containing words and bounding boxes.
        tokenizer (object): The tokenizer used for encoding the input.

    Returns:
        list: A list of tuples containing the predicted bounding boxes and labels.
    """
    device = "cpu"
    label_list = config.class_list
    words, bboxes = convert_line2word_bbox(data)
    encoding = tokenizer(
        text=words,
        boxes=bboxes,
        padding="max_length",
        truncation=True,
        max_length=config.max_token_len,  # Fixed to 512 tokens
        return_tensors="pt",
        return_overflowing_tokens=True,
        stride=config.stride,  # Overlap between chunks
        return_offsets_mapping=True)
    encoding.pop("offset_mapping")
    encoding.pop("overflow_to_sample_mapping")
    encoding = {k: v.type(torch.long).to(device)
                if k != 'bbox' else v.to(device) for k, v in encoding.items()}
    offset = torch.frac(
        encoding["bbox"]) if encoding["bbox"].dtype == torch.float else torch.zeros_like(encoding["bbox"])
    encoding["bbox"] = encoding["bbox"].long()
    outputs = run_onnx_model(inference_model,
                             [encoding["input_ids"], encoding["bbox"], encoding["attention_mask"]])
    # outputs = run_torch_model(inference_model, encoding)
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
    return final_pred_list


def predict(data, model):
    # Perform inference
    predictions = []
    inference_model, tokenizer = model
    file_name = os.path.basename(data).split(".")[0]
    data = extract_text_and_bbox_from_pdf(data)
    for page_num, page in enumerate(data):
        img_cv = page["image"]
        text_data = page["text"]
        page_prediction = predict_page(inference_model, text_data,
                                       tokenizer)
        visualize_predictions(img_cv, page_prediction,
                              f'{file_name}-{page_num}')
        predictions.append(page_prediction)

    return predictions


if __name__ == "__main__":
    from pdf_processor import extract_text_and_bbox_from_pdf
    model, tokenizer = model_load(
        "/mnt/c/Users/Hrishikesh/Desktop/hrishi/WORK/RESEARCH/show/LILT/")
    data = "Shi_Multi-Modal_Multi-Action_Video_Recognition_ICCV_2021_paper - Adars.pdf"
    predict(data, (model, tokenizer))
