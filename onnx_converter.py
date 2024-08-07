import torch
import os
from typing import Tuple, Dict
from onnxruntime.quantization import quantize_dynamic
from config import logger
# Assuming these imports are from your project structure
from config import TrainConfig
from inference import load_model


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


def main():
    device = "cpu" if torch.cuda.is_available() else "cpu"
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
    main()
