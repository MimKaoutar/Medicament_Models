import numpy as np 
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from models import LogisticRegression_Simple
from models import LogisticRegression_Penalty
from models import NeuralNetwork


def train_optimize(conf, device, df_sampler_temp, df_sampler, num_epochs, activation, layers, hidden_dim_1, hidden_dim_2, learning_rate, learning_rate_penalty):
    loss_fn = nn.BCELoss()
    kf = KFold(n_splits=conf.n_splits, random_state=conf.sample_random_state, shuffle=True)
    losses_val = []
    #df_risk_temp = pd.concat([X, y], axis=1)
    #df_risk_unique = df_risk_temp.drop_duplicates()
    #X= df_risk_unique.iloc[:,:-1]
    #y= df_risk_unique.iloc[:,-1]
    #df_risk_temp.drop(columns=['risk'], inplace=True)
    for train_idx, val_idx in kf.split(X=df_sampler_temp):

        X_sampler_train = df_sampler_temp.iloc[train_idx]
        X_sampler_val = df_sampler_temp.iloc[val_idx]

        df_train = df_sampler[df_sampler['hash'].isin(X_sampler_train['hash'])].drop(columns=['hash'])
        df_val = df_sampler[df_sampler['hash'].isin(X_sampler_val['hash'])].drop(columns=['hash'])
        
        #drop duplicates of test 
        #xtest = xtest.drop_duplicates()
        #ytest = ytest[xtest.index]

        xtrain = df_train.iloc[:,:-2]
        ytrain = df_train.iloc[:,-1]
        xval = df_val.iloc[:,:-2]
        yval = df_val.iloc[:,-2]
        
        
        num_inputs, num_features = xtrain.shape[0], xtrain.shape[1]
        xtrain = xtrain.to_numpy(dtype="float32")
        ytrain = ytrain.to_numpy(dtype="float32")
        xval = xval.to_numpy(dtype="float32")
        yval = yval.to_numpy(dtype="float32")
        #tranform to tensor 
        X_train_tensor = torch.tensor(xtrain, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(ytrain, dtype=torch.float32).unsqueeze(1).to(device)  # Add a dimension for compatibility
        X_val_tensor = torch.tensor(xval, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(yval, dtype=torch.float32).unsqueeze(1).to(device)


        if conf.model == 'Logistic_regression':
            model = LogisticRegression_Simple(device, num_features=num_features)
            optimizer = torch.optim.Adam(model.parameters(), eps=1e-08, lr = learning_rate)
        if conf.model == 'Logistic_regression_penalty':
            model = LogisticRegression_Penalty(device, num_features=num_features, num_inputs=num_inputs)
            # Optimizer for weights and biases (main model parameters)
            optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if name != "penalties"], eps=1e-08, lr=learning_rate)
            # Optimizer for penalties (separate learning rate) + different direction
            optimizer_penalty = torch.optim.Adam([model.penalties], lr=learning_rate_penalty, maximize=True)
        if conf.model == 'Neural_network' and conf.mode == 0:
            model = NeuralNetwork(device, activation=activation, num_features=num_features,mode=0, layers=layers, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2)
            optimizer = torch.optim.Adam(model.parameters(), eps=1e-08, lr = learning_rate)
        if conf.model == 'Neural_network' and conf.mode == 1:
            model = NeuralNetwork(device, activation=activation, num_features=num_features,mode=1, layers=layers, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2)
            optimizer = torch.optim.Adam(model.parameters(), eps=1e-08, lr = learning_rate)
            optimizer_penalty = torch.optim.Adam([model.penalties], lr=learning_rate_penalty, maximize=True)
        
        model.to(device)
        for epoch in range(num_epochs):
            # Forward pass
            y_pred_train = model(X_train_tensor)
            
            # Compute loss
            loss = loss_fn(y_pred_train , y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer.step()
            if conf.model == 'Logistic_regression_penalty':
                optimizer_penalty.step()
            if conf.model == 'Neural_network' and conf.mode == 1:
                optimizer_penalty.step()

        y_pred_val = model(X_val_tensor)
        loss = loss_fn(y_pred_val , y_val_tensor)
        losses_val.append(loss.item())
    return np.mean(losses_val)

#objective function
def optimize(trial,conf, device,  df_sampler_temp, df_sampler):
    # Hyperparameter suggestions using Optuna
    num_epochs = trial.suggest_categorical("epoch",[100,500,1000,2000,5000,10000]) 
    activation = None
    layers = None
    hidden_dim_1 = None
    hidden_dim_2 = None
    learning_rate_penalty = None
    if conf.model == 'Neural_network':
        activation = trial.suggest_categorical("activation", ['RELU', 'LRELU'])
        layers = trial.suggest_categorical("layers", [1,2])
        hidden_dim_1 = trial.suggest_categorical("hidden_dim_1", [100,200,300,400,500])
        if layers == 2 : 
            hidden_dim_2 = trial.suggest_categorical("hidden_dim_2", [100,200,300,400,500])
        if conf.mode == 1: 
            learning_rate_penalty = trial.suggest_float("learning_rate_penalty", 1e-4, 1e-1, log=True)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)  # Regularization strength
    if conf.model == "Logistic_regression_penalty":
        learning_rate_penalty = trial.suggest_float("learning_rate_penalty", 1e-4, 1e-1, log=True)
    
    loss = train_optimize(conf, device, df_sampler_temp, df_sampler, num_epochs, activation, layers, hidden_dim_1, hidden_dim_2, learning_rate, learning_rate_penalty)
    return loss