import pandas as pd 
import os
import torch 
from torch import nn
import lightning as L   
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt


# Hyper-parameters

num_epochs = 10
batch_size = 16
learning_rate = 0.001

class NeuralNetwork(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(304,1)
        self.relu = nn.ReLU() 
        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self,x):
        x = self.linear(x)
        return x
    
    def train_dataloader(self):
        
        df = pd.read_csv('train.csv')
        X_train = df.drop(columns=['diagnosis'])
        y_train = df['diagnosis']
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size,shuffle=False
        )
        return train_loader

    def test_dataloader(self):

        df = pd.read_csv('test.csv')
        X_test = df.drop(columns=['diagnosis'])
        y_test = df['diagnosis']
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )
        return test_loader
    
    def configure_optimizers(self):
        return SGD(self.parameters(), lr= learning_rate)
    
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self(input_i)
        loss = self.loss(output_i.view(-1), label_i)
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss, prog_bar=True)
        return {"loss": loss, 'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_idx):
        input_i, label_i = batch
        outputs = self(input_i)
        loss = self.loss(outputs.view(-1), label_i) 
        self.validation_step_outputs.append(loss)    
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.log('avg_val_loss',avg_loss, prog_bar=True)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    
if __name__ == '__main__':

    model = NeuralNetwork()
    trainer = L.Trainer(max_epochs=20, accelerator="auto", devices="auto")
    trainer.fit(model)
    trainer.validate(model)
    

