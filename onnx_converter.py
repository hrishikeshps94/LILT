import torch
import os
from typing import Tuple, Dict
from onnxruntime.quantization import quantize_dynamic
from config import logger
# Assuming these imports are from your project structure
from config import TrainConfig
from inference import load_model, run_onnx_model
import onnxruntime
from eval import calculate_metrics
import tqdm
from data import CustomDataset, collate_fn, BatchSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def create_dummy_inputs(batch_size: int = 1, seq_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy inputs for the model."""
    input_ids = torch.randint(0, 50265, (batch_size, seq_length)).long()
    bbox = torch.cat((
        torch.randint(0, 500, (batch_size, seq_length, 2)).long(),
        torch.randint(500, 1000, (batch_size, seq_length, 2)).long()
    ), dim=-1)
    attention_mask = torch.randint(0, 1, (batch_size, seq_length)).long()
    return input_ids, bbox, attention_mask


def get_dynamic_axes() -> Dict[str, Dict[int, str]]:
    """Define dynamic axes for ONNX export."""
    symbolic_names = {0: 'batch_size', 1: 'sequence'}
    return {
        'input_ids': symbolic_names,
        'bbox': symbolic_names,
        'attention_mask': symbolic_names,
        'output': symbolic_names
    }


def onnx_convert(model: torch.nn.Module, onnx_path: str):
    """Convert PyTorch model to ONNX format."""
    model.eval()
    dummy_inputs = create_dummy_inputs()

    try:
        torch.onnx.export(
            model,
            args=dummy_inputs,
            f=onnx_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_ids', 'bbox', 'attention_mask'],
            output_names=['output'],
            dynamic_axes=get_dynamic_axes()
        )
        logger.info(
            f"Model successfully converted to ONNX. Saved at: {onnx_path}")
    except Exception as e:
        logger.error(f"Error during ONNX conversion: {str(e)}")
        raise


def quantize_onnx_model(input_path: str, output_path: str):
    """Quantize the ONNX model."""
    try:
        quantize_dynamic(model_input=input_path, model_output=output_path)
        logger.info(f"Model successfully quantized. Saved at: {output_path}")
    except Exception as e:
        logger.error(f"Error during model quantization: {str(e)}")
        raise


def val_dataset(tokenizer):
    """
    Create a validation dataset loader.

    Args:
        tokenizer (Tokenizer): The tokenizer object to use for tokenizing the data.

    Returns:
        DataLoader: The validation dataset loader.

    """
    val_dataset = CustomDataset(TrainConfig.data_path, tokenizer, mode='test')
    val_sampler = BatchSampler(
        val_dataset, batch_size=1, keep_last_batch=True, shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)
    return val_loader


def evaluate(model_path, val_loader):
    """
    Evaluate the model on the validation dataset.

    Args:
        model_path (str): The path to the model file.
        val_loader (torch.utils.data.DataLoader): The validation data loader.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    logger.info("Evaluating the model on the validation dataset...")
    mode = "torch"
    if model_path.endswith(".onnx"):
        mode = "onnx"
        model = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
    if model_path.endswith(".pt"):
        model, _ = load_model(model_path)
        model = model.to(device)
        model.eval()
    final_predictions = {}
    for idx, batch in enumerate(tqdm.tqdm(val_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        if mode == "onnx":
            outputs = run_onnx_model(model,
                                     [batch["input_ids"], batch["bbox"], batch["attention_mask"]])
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
        else:
            with torch.no_grad():
                outputs = model(**batch, return_dict=True)
                outputs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = outputs.argmax(-1)
        final_predictions.setdefault('predictions', []).extend(predictions)
        final_predictions.setdefault('labels', []).extend(batch["labels"])
    results = calculate_metrics(
        torch.cat(final_predictions['predictions']), torch.cat(final_predictions['labels']), TrainConfig.class_list)
    return results


def performance_analyser():
    """
    Analyzes the performance of different models by evaluating them on a validation dataset.

    This function evaluates the performance of three models: the original model, the ONNX model, and the quantized ONNX model.
    It uses the `evaluate` function to evaluate each model on a validation dataset and logs the results.

    Args:
        None

    Returns:
        None
    """
    org_model_path = 'weights/best.pt'
    onnx_model_path = 'weights/best.onnx'
    quantized_model_path = 'weights/quantized_best.onnx'
    models = [org_model_path, onnx_model_path, quantized_model_path]
    tokenizer = AutoTokenizer.from_pretrained(
        TrainConfig.base_model_name)
    val_loader = val_dataset(tokenizer)
    for model_path in models:
        logger.info(f"Evaluating model: {model_path}")
        result = evaluate(model_path, val_loader)
        logger.info(f"Model results: {result}")


def main():
    try:
        # Load the model
        model, _ = load_model(model_path='weights/best.pt')
        model = model.to(device)

        # Convert to ONNX
        onnx_path = os.path.join(TrainConfig.onnx_path, 'best.onnx')
        onnx_convert(model, onnx_path)

        # Quantize the model
        quantized_path = os.path.join(
            TrainConfig.onnx_path, 'quantized_best.onnx')
        quantize_onnx_model(onnx_path, quantized_path)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    device = "cpu" if torch.cuda.is_available() else "cpu"
    main()
    performance_analyser()
