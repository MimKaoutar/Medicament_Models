from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pandas as pd 
import os
import torch 
from torch import nn
import lightning as L   
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from torch.optim.lr_scheduler import StepLR
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

a = pd.Series(list(mpl.rcParams.keys()))
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.titlesize'] = 12

# Hyper-parameters

select = False
batch_size = 64
learning_rate = 0.1

class NeuralNetwork(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.linear = nn.Linear(106,1)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.lr = 0.1
        self._initialize_weights()
        bias_value = -2.197
        nn.init.constant_(self.linear.bias, bias_value)
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.confmat = torchmetrics.ConfusionMatrix(task="binary")
        self.prcurve = torchmetrics.PrecisionRecallCurve(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.roccurve = torchmetrics.ROC(task="binary")
        #self.f1_score = torchmetrics.F1Score(task="binary")

        collection = torchmetrics.MetricCollection([
            self.accuracy,
            self.confmat,
            self.precision,
            self.recall,
            self.roccurve,
            self.prcurve
        ])

        train_metrics = collection.clone(prefix='train_')
        val_metrics = collection.clone(prefix='val_')
        test_metrics = collection.clone(prefix='test_')
        self.train_tracker = torchmetrics.wrappers.MetricTracker(train_metrics)
        self.val_tracker = torchmetrics.wrappers.MetricTracker(val_metrics)
        self.test_tracker = torchmetrics.wrappers.MetricTracker(test_metrics)

        self.predictions = []
        self.true_labels = []
        self.val_predictions = []
        self.val_true_labels = []
       
        

    def _initialize_weights(self, seed=42):
        # Xavier initialization for linear layer
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x_out = self.linear(x)
        x = self.dropout(x_out)
        return x
    
    def train_dataloader(self):
        df = pd.read_csv('train.csv')
        if select:

            df1 = df[df['diagnosis'] == 0].iloc[:10, :]
            df2 = df[df['diagnosis'] == 1].iloc[:10, :]
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
    
    def val_dataloader(self):
        df = pd.read_csv('val.csv')
        if select:

            df1 = df[df['diagnosis'] == 0].iloc[:10, :]
            df2 = df[df['diagnosis'] == 1].iloc[:10, :]
            df = pd.concat([df1, df2], axis=0)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            #batch_size = 5

        X_val = df.drop(columns=['diagnosis'])
        y_val = df['diagnosis']
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=47)

        return val_loader
    
    
    def test_dataloader(self) :
        df = pd.read_csv('test.csv')
        if select:
            df = df.iloc[:100, :]
            #batch_size = 1
        X_test = df.drop(columns=['diagnosis'])
        y_test = df['diagnosis']
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return  test_loader
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        input= batch[0]
        true_label = batch[1]
        output = self(input)
        train_loss = self.loss(output.view(-1), true_label)
        probabilities = torch.sigmoid(output).squeeze(-1)
        predicted_label = (probabilities > 0.5).long()
        acc = self.accuracy(predicted_label, true_label)
        self.log('train_loss', train_loss, on_epoch=True, sync_dist =True)
        self.log("train_accuracy", acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist =True)
        
        self.predictions.append(predicted_label)
        self.true_labels.append(true_label)
    
        return train_loss
    
    def on_train_epoch_end(self):
        all_preds = torch.cat(self.predictions, dim=0).float()
        all_labels = torch.cat(self.true_labels, dim=0).int()

        self.train_tracker.increment()
        self.train_tracker.update(all_preds, all_labels)
        self.predictions.clear()
        self.true_labels.clear()
     
    def on_train_end(self):

        all_results = self.train_tracker.compute_all()
        print(all_results)

        fig1 = plt.figure(figsize=(10,6))
        ax1 = plt.axes()

        

        # ConfusionMatrix and ROC we just plot the last step, notice how we call the plot method of those metrics
        self.confmat.plot(val=all_results[-1]['train_BinaryConfusionMatrix'], ax=ax1, )

        fig2 = plt.figure(figsize=(10,6))
        ax2 = plt.axes()
        self.prcurve.plot(all_results[-1]["train_BinaryPrecisionRecallCurve"], ax=ax2)
       
        fig3 = plt.figure(figsize=(10,6))
        ax3 = plt.axes()
        # For the remaining we plot the full history, but we need to extract the scalar values from the results
        scalar_results = [
            {k: v for k, v in ar.items() if isinstance(v, torch.Tensor) and v.numel() == 1} for ar in all_results
        ]
        self.train_tracker.plot(val=scalar_results, ax=ax3)
       
        fig4 = plt.figure(figsize=(10,6))
        ax4 = plt.axes()
        self.roccurve.plot(all_results[-1]["train_BinaryROC"], ax=ax4)

        fig1.savefig("train_CM_Plot.png")
        fig2.savefig("train_PRcurve_Plots.png")
        fig3.savefig("train_metrics_Plots.png")
        fig4.savefig("train_roccurve_Plots.png")
  
    def validation_step(self, batch, batch_idx):
        input,true_label = batch
        output = self(input)
        val_loss = self.loss(output.view(-1), true_label)
        probabilities = torch.sigmoid(output).squeeze(-1)
        predicted_label = (probabilities > 0.5).long()
        acc = self.accuracy(predicted_label, true_label)
        self.log('val_loss', val_loss, prog_bar=True,  sync_dist =True)
        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist =True)
        self.val_predictions.append(predicted_label)
        self.val_true_labels.append(true_label)
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_predictions, dim=0).float()
        all_labels = torch.cat(self.val_true_labels, dim=0).int()

        self.val_tracker.increment()
        self.val_tracker.update(all_preds, all_labels)
        self.val_predictions.clear()
        self.val_true_labels.clear()

    def on_validation_end(self):

        all_results = self.val_tracker.compute_all()
        fig1 = plt.figure(figsize=(10,6))
        ax1 = plt.axes()

        

        # ConfusionMatrix and ROC we just plot the last step, notice how we call the plot method of those metrics
        self.confmat.plot(val=all_results[-1]['val_BinaryConfusionMatrix'], ax=ax1, )

        fig2 = plt.figure(figsize=(10,6))
        ax2 = plt.axes()
        self.prcurve.plot(all_results[-1]["val_BinaryPrecisionRecallCurve"], ax=ax2)
       
        fig3 = plt.figure(figsize=(10,6))
        ax3 = plt.axes()
        # For the remaining we plot the full history, but we need to extract the scalar values from the results
        scalar_results = [
            {k: v for k, v in ar.items() if isinstance(v, torch.Tensor) and v.numel() == 1} for ar in all_results
        ]
        self.val_tracker.plot(val=scalar_results, ax=ax3)
       
        fig4 = plt.figure(figsize=(10,6))
        ax4 = plt.axes()
        self.roccurve.plot(all_results[-1]["val_BinaryROC"], ax=ax4)

        fig1.savefig("val_CM_Plot.png")
        fig2.savefig("val_PRcurve_Plots.png")
        fig3.savefig("val_metrics_Plots.png")
        fig4.savefig("val_roccurve_Plots.png")

    def test_step(self, batch, batch_idx):
        input,true_label = batch
        output = self(input)
        probabilities = torch.sigmoid(output).squeeze(-1)
        predicted_label = (probabilities > 0.5).long()
        acc = self.accuracy(predicted_label, true_label)
        self.log("test_accuracy", acc, sync_dist=True)
        self.predictions.append(predicted_label)
        self.true_labels.append(true_label)
    
    
    
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.predictions, dim=0).float()
        all_labels = torch.cat(self.true_labels, dim=0).int()

        self.test_tracker.increment()
        self.test_tracker.update(all_preds, all_labels)
        self.predictions.clear()
        self.true_labels.clear()

        #self.test_tracker.compute()
        all_results = self.test_tracker.compute()

        fig1 = plt.figure(figsize=(10,6))
        ax1 = plt.axes()
        

        # ConfusionMatrix and ROC we just plot the last step, notice how we call the plot method of those metrics
        self.confmat.plot(val=all_results['test_BinaryConfusionMatrix'], ax=ax1, )

        fig2 = plt.figure(figsize=(10,6))
        ax2 = plt.axes()
        self.prcurve.plot(all_results["test_BinaryPrecisionRecallCurve"], ax=ax2)
       
        fig3 = plt.figure(figsize=(10,6))
        ax3 = plt.axes()
        # For the remaining we plot the full history, but we need to extract the scalar values from the results
        scalar_results = [
            {k: v for k, v in all_results.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
        ]
        print(scalar_results)

        self.prcurve.plot(all_results["test_BinaryROC"], ax=ax3)
       
        fig1.savefig("test_CM_Plot.png")
        fig2.savefig("test_PRcurve_Plots.png")
        fig3.savefig("test_ROC_Plots.png")
        

checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

tb_logger = TensorBoardLogger(save_dir='lightning_logs', name='model')
model = NeuralNetwork()
trainer = Trainer(max_epochs=50,
                devices=1,
                accelerator="gpu",
                strategy="ddp",
                logger=tb_logger,
                log_every_n_steps = 1,
                fast_dev_run=False,
                callbacks=[checkpoint_callback])

model.lr = 3e-4
trainer.fit(model)
trainer.test(model)
