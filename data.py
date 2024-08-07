import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from transformers import LayoutLMv3TokenizerFast
from config import TrainConfig
import tqdm
import numpy as np
from utils import normalize_bbox, xywh2xyxy


class CustomDataset(Dataset):
    """
    Custom dataset class for processing data for training or evaluation.

    Args:
        folder_path (str): The path to the folder containing the data.
        tokenizer (Tokenizer): The tokenizer object used for tokenizing the text.
        mode (str, optional): The mode of the dataset. Defaults to 'train'.

    Attributes:
        annotation_folder_path (str): The path to the annotations folder.
        annotation_file_paths (list): A list of paths to the annotation files.
        tokenizer (Tokenizer): The tokenizer object used for tokenizing the text.
        label_map (dict): A dictionary mapping labels to their corresponding indices.
        chunks (list): A list of preprocessed data chunks.

    Methods:
        preprocess_data: Preprocesses the data by processing each annotation file.
        process_file: Processes a single annotation file.
        __len__: Returns the length of the dataset.
        __getitem__: Returns a data chunk at the specified index.
    """

    def __init__(self, folder_path, tokenizer, mode='train'):
        self.config = TrainConfig()
        self.mode = mode
        self.annotation_folder_path = os.path.join(
            folder_path, mode, 'annotations')
        self.annotation_file_paths = sorted(
            [f"{self.annotation_folder_path}/{file}" for file in os.listdir(self.annotation_folder_path)])
        self.tokenizer = tokenizer
        self.label_map = {label: i for i,
                          label in enumerate(self.config.class_list)}
        self.chunks = self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocesses the data by processing each annotation file.

        Returns:
            list: A list of preprocessed data chunks.
        """
        all_chunks = []
        for file_path in tqdm.tqdm(self.annotation_file_paths, desc=f"Processing data: {self.mode}"):
            doc_chunks = self.process_file(file_path)
            all_chunks.extend(doc_chunks)
        return all_chunks

    def process_file(self, file_path):
        """
        Processes a single annotation file.

        Args:
            file_path (str): The path to the annotation file.

        Returns:
            list: A list of data chunks extracted from the annotation file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        sentences, bboxes, labels = [], [], []
        for item in data["form"]:
            sentences.append(item["text"])
            labels.append(self.label_map[item["category"]])
            bboxes.append(normalize_bbox(
                xywh2xyxy(item["box_line"]), 1025, 1025))

        encoding = self.tokenizer(
            text=sentences,
            boxes=bboxes,
            word_labels=labels,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_token_len,  # Fixed to 512 tokens
            return_tensors="pt",
            return_overflowing_tokens=True,
            stride=self.config.stride,  # Overlap between chunks
            return_offsets_mapping=True
        )

        chunks = []
        for i in range(len(encoding['input_ids'])):
            chunk = {
                "input_ids": encoding["input_ids"][i],
                "attention_mask": encoding["attention_mask"][i],
                "bbox": encoding["bbox"][i],
                "labels": encoding["labels"][i],
                "file_path": file_path,
                "chunk_index": i
            }
            chunks.append(chunk)
        return chunks

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Returns a data chunk at the specified index.

        Args:
            idx (int): The index of the data chunk.

        Returns:
            dict: A data chunk containing input_ids, attention_mask, bbox, labels, file_path, and chunk_index.
        """
        return self.chunks[idx]


def collate_fn(batch):
    """
    Collate function for batching data.

    Args:
        batch (list): A list of dictionaries, where each dictionary represents a single sample.

    Returns:
        dict: A dictionary containing the batched data, with the following keys:
            - "input_ids": A tensor of shape (batch_size, max_seq_length) representing the input IDs.
            - "attention_mask": A tensor of shape (batch_size, max_seq_length) representing the attention mask.
            - "bbox": A tensor of shape (batch_size, max_seq_length, 4) representing the bounding boxes.
            - "labels": A tensor of shape (batch_size, max_seq_length) representing the labels.
            - "file_paths": A list of file paths.
            - "chunk_indices": A list of chunk indices.
    """
    input_ids = torch.stack([item["input_ids"]
                            for item in batch]).type(torch.long)
    attention_mask = torch.stack([item["attention_mask"]
                                 for item in batch]).type(torch.long)
    bbox = torch.stack([item["bbox"] for item in batch]).type(torch.long)
    labels = torch.stack([item["labels"] for item in batch]).type(torch.long)
    # file_paths = [item["file_path"] for item in batch]
    # chunk_indices = [item["chunk_index"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bbox,
        "labels": labels,
        # "file_paths": file_paths,
        # "chunk_indices": chunk_indices
    }


class BatchSampler(torch.utils.data.Sampler):
    """
    A custom batch sampler that generates batches of indices for a given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        batch_size (int): The size of each batch.
        keep_last_batch (bool, optional): Whether to keep the last batch if its size is less than `batch_size`. 
            Defaults to False.
        shuffle (bool, optional): Whether to shuffle the indices before generating batches. Defaults to False.
    """

    def __init__(self, dataset, batch_size, keep_last_batch=False, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + self.batch_size -
                            1) // self.batch_size if keep_last_batch else self.num_samples // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples).tolist()
        else:
            indices = np.arange(self.num_samples)
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_samples)
            yield indices[start_idx:end_idx]

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":
    data_config = TrainConfig()
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        "microsoft/layoutlmv3-base")
    tokenizer.sep_token_box = data_config.tok_bbox_sep
    tokenizer.pad_token_box = data_config.tok_bbox_sep

    train_dataset = CustomDataset("dataset", tokenizer, mode='train')

    sampler = BatchSampler(
        train_dataset, batch_size=data_config.batch_size, keep_last_batch=True, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        # num_workers=16, pin_memory=True
    )
    for _ in range(100):
        for batch in tqdm.tqdm(train_loader):
            tokenizer.decode(batch["input_ids"][0])
