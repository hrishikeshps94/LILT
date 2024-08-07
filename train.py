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

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')


# Calling train configurations
# Loading from ther pretrained model
# Load the config of the model from the author repository
config = LiltConfig.from_pretrained(TrainConfig.base_model_name)
config.num_labels = len(TrainConfig.class_list)
model = TokenClassificationModel(config)


# Load the model from the author huggingface hub
hugg_model = LiltForTokenClassification.from_pretrained(
    TrainConfig.base_model_name)
# Copy the model state from the huggingface model to the custom model
copy_model_state(model, hugg_model)

# Load the tokenizer from author repository this
tokenizer = AutoTokenizer.from_pretrained(TrainConfig.base_model_name)
# LayoutLMv3 tokenizer , you can also call the tokenizer with LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
# Loading data from the custom dataset

train_dataset = CustomDataset(TrainConfig.data_path, tokenizer, mode='train')
val_dataset = CustomDataset(TrainConfig.data_path, tokenizer, mode='val')
test_dataset = CustomDataset(TrainConfig.data_path, tokenizer, mode='test')

# Creating a batch sampler
train_sampler = BatchSampler(
    train_dataset, batch_size=TrainConfig.batch_size, keep_last_batch=True, shuffle=True)
val_sampler = BatchSampler(
    val_dataset, batch_size=TrainConfig.batch_size, keep_last_batch=True, shuffle=False)
test_sampler = BatchSampler(
    test_dataset, batch_size=TrainConfig.batch_size, keep_last_batch=True, shuffle=False)

# Creating a dataloader
train_loader = DataLoader(
    train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(
    val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)
test_loader = DataLoader(
    test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

# Training the model
optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.max_lr)
model = model.to(device)
# model = torch.compile(model)


get_lr = partial(get_lr, config=TrainConfig,
                 iterations_per_epoch=len(train_loader))
step_count = 0
best_f1 = 0
for epoch in range(TrainConfig.epochs):
    print("Epoch:", epoch+1)
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
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step_count)
        step_count += 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        logger.info(
            f"Step: {step_count} | Time taken: {(t1-t0)*1000:.4f} ms| LR: {lr:.6f} | Norm: {norm:.2f} | Loss: {loss.item():.2f}")
    if epoch % TrainConfig.save_epochs == 0:
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
        if results["overall_f1"] > best_f1:
            best_f1 = results["overall_f1"]
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': epoch,
                'f1_score': best_f1
            }
            os.makedirs(TrainConfig.save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(
                TrainConfig.save_path, 'best.pt'))

        logger.info(
            f'Overall f1: {results["overall_f1"]} | Overall Precision: {results["overall_precision"]} | Overall Recall: {results["overall_recall"]}')
