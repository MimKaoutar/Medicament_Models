import pandas as pd 
import os
import torch 
from torch import nn
import lightning as L   
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torchmetrics import Accuracy
import seaborn as sns
import matplotlib.pyplot as plt
from torcheeg.model_selection import KFold
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

k_fold = KFold(n_splits=10,
                      split_path=f'./tmp_out/split',
                      shuffle=True,
                      random_state=42)


# Hyper-parameters

batch_size = 16
learning_rate = 0.001

class NeuralNetwork(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.linear = nn.Linear(304,1)
        self.relu = nn.ReLU() 
        self.loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

    def forward(self,x):
        x = self.linear(x)
        return x
    
    def configure_optimizers(self):
        return SGD(self.parameters(), lr= learning_rate)
    
    def training_step(self, batch, batch_idx):
        input, true_label = batch
        output = self(input)
        train_loss = self.loss(output.view(-1), true_label)
        predicted_label = torch.sigmoid(output)
        acc = self.train_accuracy(predicted_label, true_label)
        self.log('train_loss', train_loss)
        self.log("train_accuracy", acc, prog_bar=True, on_epoch=True, on_step=False)
        return train_loss
       

    def validation_step(self, batch, batch_idx):
        input, true_label = batch
        output = self(input)
        val_loss = self.loss(output.view(-1), true_label)
        predicted_label = torch.sigmoid(output)
        acc = self.train_accuracy(predicted_label, true_label)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        input, true_label = batch
        output = self(input)
        predicted_label = torch.sigmoid(output)
        acc = self.train_accuracy(predicted_label, true_label)
        self.log("test_accuracy", acc)
        
df = pd.read_csv('train.csv')
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
X_tensor = torch.tensor(X.values, dtype=torch.float)
y_tensor = torch.tensor(y.values, dtype=torch.float)
dataset = TensorDataset(X_tensor, y_tensor)

df = pd.read_csv('test.csv')
X_test = df.drop(columns=['diagnosis'])
y_test = df['diagnosis']
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        tb_logger = TensorBoardLogger(save_dir='lightning_logs', name=f'fold_{i + 1}')
        checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                              filename="{epoch:02d}-{val_metric:.4f}",
                                              monitor='val_metric',
                                              mode='max')

        model = NeuralNetwork()

        trainer = Trainer(max_epochs=50,
                          devices=2,
                          accelerator="auto",
                          strategy="ddp",
                          checkpoint_callback=checkpoint_callback,
                          logger=tb_logger,
                          fast_dev_run=True)

        trainer.fit(model, train_loader, val_loader)



                






