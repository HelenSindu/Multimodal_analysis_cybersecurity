import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm

from model.config import *
from model.dataset import MemoryEfficientDataset
from model.multimodal_model import LiteDNN, LiteCNN, LiteBERT, OptimizedMultimodalModel
from model.train import train_epoch, evaluate
from model import prepare_tabular_data,prepare_image_data, prepare_sequence_data

tabular_data_train = pd.read_csv('train-tabular.csv')
tabular_data_test = pd.read_csv('test-tabular.csv')
sequence_data_train = pd.read_json('train-sequence.json')
sequence_data_test = pd.read_json('test-sequence.json')

print("Preparing tabular data...")
tabular_dataset_train, tabular_dataset_test = prepare_tabular_data(tabular_data_train, tabular_data_test)
print("Finished!")
print("Preparing image data...")
image_dataset_train, image_dataset_test = prepare_image_data('/content/Image/train', '/content/Image/test')
print("Finished!")
print("Preparing sequence data...")
sequence_dataset_train, sequence_dataset_test = prepare_sequence_data(sequence_data_train, sequence_data_test)
print("Finished!")

train_dataset = MemoryEfficientDataset(tabular_dataset_train, image_dataset_train, sequence_dataset_train)
test_dataset = MemoryEfficientDataset(tabular_dataset_test, image_dataset_test, sequence_dataset_test)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True)

    # Initialize model
model = OptimizedMultimodalModel(
    tabular_input_dim=len(tabular_dataset_train[0]['apk_features']),
    num_classes=NUM_CLASSES).to(DEVICE)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
scaler = torch.cuda.amp.GradScaler()


for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if (epoch + 1) % 3 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss},
            f"checkpoint_epoch_{epoch+1}.pth")

        torch.cuda.empty_cache()
        gc.collect()
