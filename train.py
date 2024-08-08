from data import CustomDataset, collate_fn, BatchSampler
from transformers import AutoTokenizer
from model import TokenClassificationModel
from transformers.models.lilt import LiltConfig
from transformers import LiltForTokenClassification
import torch
import random
import numpy as np
from transformers import set_seed
from config import TrainConfig
from torch.utils.data import DataLoader
from utils import get_lr, copy_model_state
from functools import partial
import os
import time
from config import logger
import tqdm
from eval import calculate_metrics
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setting the seed for reproducibility
seed = 42
# Set seed for Python random module
random.seed(seed)
# Set seed for NumPy
np.random.seed(seed)
# Set seed for Torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision('high')


def model_init():
    """
    Initializes and returns a TokenClassificationModel for training.

    Returns:
        model (TokenClassificationModel): The initialized TokenClassificationModel.
    """
    config = LiltConfig.from_pretrained(TrainConfig.base_model_name)
    config.num_labels = len(TrainConfig.class_list)
    model = TokenClassificationModel(config)
    tokenizer = AutoTokenizer.from_pretrained(TrainConfig.base_model_name)
    return model, tokenizer


def load_dataset(tokenizer):
    """
    Load the dataset for training.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the dataset.

    Returns:
        train_loader (DataLoader): The DataLoader for the training dataset.
        val_loader (DataLoader): The DataLoader for the validation dataset.
        test_loader (DataLoader): The DataLoader for the test dataset.
    """
    train_dataset = CustomDataset(
        TrainConfig.data_path, tokenizer, mode='test')
    val_dataset = CustomDataset(TrainConfig.data_path, tokenizer, mode='val')
    test_dataset = CustomDataset(TrainConfig.data_path, tokenizer, mode='test')
    train_sampler = BatchSampler(
        train_dataset, batch_size=TrainConfig.batch_size, keep_last_batch=True, shuffle=True)
    val_sampler = BatchSampler(
        val_dataset, batch_size=TrainConfig.batch_size, keep_last_batch=True, shuffle=False)
    test_sampler = BatchSampler(
        test_dataset, batch_size=TrainConfig.batch_size, keep_last_batch=True, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def optimizer_scheduler_init(model, train_loader):
    """
    Initializes the optimizer and learning rate scheduler for training.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.

    Returns:
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        get_lr (callable): A function that returns the learning rate for each training iteration.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.max_lr)
    scheduler = partial(get_lr, config=TrainConfig,
                        iterations_per_epoch=len(train_loader))
    return optimizer, scheduler


def evaluate(model, val_loader):
    """
    Evaluates the model on the validation data loader.

    Args:
        model (TokenClassificationModel): The model to evaluate.
        val_loader (DataLoader): The DataLoader for the validation dataset.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    logger.info("Evaluating the model on the validation dataset...")
    model.eval()
    final_predictions = {}
    for idx, batch in enumerate(tqdm.tqdm(val_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, return_dict=True)
        predictions = outputs.logits.argmax(-1)
        final_predictions.setdefault('predictions', []).extend(predictions)
        final_predictions.setdefault('labels', []).extend(batch["labels"])
    results = calculate_metrics(
        torch.cat(final_predictions['predictions']), torch.cat(final_predictions['labels']), TrainConfig.class_list)
    return results


def train(model, train_loader, optimizer, scheduler):
    """
    Trains the given model using the provided data loader, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        scheduler (callable): The learning rate scheduler.

    Returns:
        None
    """
    step_count = 0
    for idx, batch in enumerate(train_loader):
        # move batch to device
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(**batch, return_dict=True)
        loss = outputs.loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        lr = scheduler(step_count)
        step_count += 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        logger.info(
            f"Step: {step_count} | Time taken: {(t1-t0)*1000:.4f} ms| LR: {lr:.6f} | Norm: {norm:.2f} | Loss: {loss.item():.2f}")


def run():
    """
    Runs the training process for the LILT model.

    This function initializes the model and tokenizer, loads the model from the huggingface hub,
    copies the model state, loads the dataset, initializes the optimizer and scheduler,
    trains the model for the specified number of epochs, evaluates the model periodically,
    saves the best model checkpoint, and logs the training progress.

    Returns:
        None
    """
    model, tokenizer = model_init()
    # Load the model from the author huggingface hub
    hugg_model = LiltForTokenClassification.from_pretrained(
        TrainConfig.base_model_name)
    # Copy the model state from the huggingface model to the custom model
    copy_model_state(model, hugg_model)
    train_loader, val_loader, _ = load_dataset(tokenizer)
    optimizer, scheduler = optimizer_scheduler_init(
        model, train_loader)

    model = model.to(device)
    model = torch.compile(model)
    best_f1 = 0
    for epoch in range(TrainConfig.epochs):
        train(model, train_loader, optimizer, scheduler)
        if epoch % TrainConfig.save_epochs == 0:
            results = evaluate(model, val_loader)
        if results["overall_f1"] > best_f1:
            best_f1 = results["overall_f1"]
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': epoch,
                'f1_score': best_f1, }
            os.makedirs(TrainConfig.save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(
                TrainConfig.save_path, 'best_test.pt'))
        logger.info(
            f'Epoch No: {epoch+1} Overall f1: {results["overall_f1"]} | Overall Precision: {results["overall_precision"]} | Overall Recall: {results["overall_recall"]}')


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("CUDA is not available. Training will be slow.")
    run()
