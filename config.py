from dataclasses import dataclass
import logging


@dataclass
class TrainConfig:
    '''Configuration for training'''
    # Training parameters
    batch_size: int = 16
    max_token_len: int = 512
    stride: int = 128
    tok_bbox_sep: tuple = (1000, 1000, 1000, 1000)
    epochs: int = 100
    warmup_epochs: int = 10
    max_lr: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    save_epochs: int = 1
    seed: int = 42
    gradient_accumulation_steps: int = 1
    base_model_name: str = 'SCUT-DLVCLab/lilt-roberta-en-base'
    data_path: str = 'dataset'
    save_path: str = 'weights'
    device: str = 'cuda'
    class_list: tuple = ("Caption", "Footnote", "Formula", "List-item", "Page-footer",
                         "Page-header", "Picture", "Section-header", "Table", "Text", "Title")
    pretrained_model_path: str = 'weights/best_exxon.pt'
    remove_labels: tuple = ()
    onnx_path: str = 'weights'


def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger('train.log')
