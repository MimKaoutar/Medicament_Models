from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pandas as pd 
import os
import torch 
from torch import nn
import lightning as L   
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from torch.optim.lr_scheduler import StepLR
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# Hyper-parameters

select = False
batch_size = 32
learning_rate =0.1

class NeuralNetwork(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.linear = nn.Linear(148,1)
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.lr = 0.1
        self._initialize_weights()
        bias_value = -1.5170
        nn.init.constant_(self.linear.bias, bias_value)
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")
        self.confmat = torchmetrics.ConfusionMatrix(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.prcurve = torchmetrics.PrecisionRecallCurve(task="binary")
        self.f1_score = torchmetrics.F1Score(task="binary")

        self.collection = torchmetrics.MetricCollection([
            self.confmat,
            self.precision,
            self.recall,
            self.prcurve,
            self.f1_score
        ])

        self.losses = []
        self.accuracies = []
        

    def _initialize_weights(self, seed=42):
        # Xavier initialization for linear layer
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)
        return x
    
    def train_dataloader(self):
        df = pd.read_csv('train.csv')
        if select:

            df1 = df[df['diagnosis'] == 0].iloc[:5, :]
            df2 = df[df['diagnosis'] == 1].iloc[:5, :]
            df = pd.concat([df1, df2], axis=0)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            #batch_size = 5

        X_train = df.drop(columns=['diagnosis'])
        y_train = df['diagnosis']
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=47)

        return train_loader
    
    
    def test_dataloader(self) :
        df = pd.read_csv('test.csv')
        if select:
            df = df.iloc[:2, :]
            #batch_size = 1
        X_test = df.drop(columns=['diagnosis'])
        y_test = df['diagnosis']
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return  test_loader
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        input= batch[0]
        true_label = batch[1]
        output = self(input)
        #print(output)
        #print(true_label)
        train_loss = self.loss(output.view(-1), true_label)
        probabilities = torch.sigmoid(output).squeeze(-1)
        predicted_label = (probabilities > 0.5).long()
        #print(probabilities)
        #print("train",train_loss)
        #print("label",predicted_label)
        acc = self.train_accuracy(predicted_label, true_label)
        self.losses.append(train_loss)
        self.accuracies.append(acc)
        self.log('train_loss', train_loss, on_epoch=True, sync_dist =True)
        self.log("train_accuracy", acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist =True)
        self.tracker.update(predicted_label, true_label)
        return train_loss
       
    """
    def validation_step(self, batch, batch_idx):
        input,true_label = batch
        output = self(input)
        val_loss = self.loss(output.view(-1), true_label)
        probabilities = torch.sigmoid(output).squeeze(-1)
        predicted_label = (probabilities > 0.5).long()
        acc = self.val_accuracy(predicted_label, true_label)
        self.log('val_loss', val_loss, prog_bar=True,  sync_dist =True)
        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist =True)
    """

    def test_step(self, batch, batch_idx):
        input,true_label = batch
        output = self(input)
        probabilities = torch.sigmoid(output).squeeze(-1)
        predicted_label = (probabilities > 0.5).long()
        acc = self.test_accuracy(predicted_label, true_label)
        self.log("test_accuracy", acc)
 
    def on_train_epoch_end(self):
        self.tracker.increment()
    
    def on_train_end(self):
        all_results = self.tracker.compute_all()

        # Plot the metrics
        fig = plt.figure(layout="constrained")
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, (3, 4))

        # ConfusionMatrix and ROC for the last step
        self.confmat.plot(val=all_results[-1]['BinaryConfusionMatrix'], ax=ax1)
        self.roc.plot(all_results[-1]["BinaryROC"], ax=ax2)

        # Plot the history for scalar metrics
        scalar_results = [
            {k: v for k, v in ar.items() if isinstance(v, torch.Tensor) and v.numel() == 1} for ar in all_results
        ]
        self.tracker.plot(val=scalar_results, ax=ax3)

        # Show plot
        plt.show()

        

tb_logger = TensorBoardLogger(save_dir='lightning_logs', name='model')
model = NeuralNetwork()
trainer = Trainer(max_epochs=50,
                devices="auto",
                accelerator="auto",
                strategy="ddp",
                logger=tb_logger,
                log_every_n_steps = 1,
                fast_dev_run=False)

model.lr = 0.1
trainer.fit(model)
torch.save(model.state_dict(), 'model.pth')
