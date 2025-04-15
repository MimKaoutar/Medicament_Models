# Standard Libraries
import random
import csv
import json

# Scientific Computing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Optimization and Hyperparameter Tuning
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
from functools import partial
from optimization import optimize

# Machine Learning Models
from models import LogisticRegression_Simple, LogisticRegression_Penalty, NeuralNetwork

# Configuration Management
from omegaconf import OmegaConf
import hydra

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import mpl_config as lmc
import shap

# Initialize Matplotlib Configuration
lmc.initialize()
plt.rcParams["text.usetex"] = False

#MUST READ :
#You should specify you output_dir if you want 
#You should add the model's best params in line 286


#get the risk of combination and save the order of columns | risk position -2 | diagnosis position -1
def get_Comb(df):
    med_df = df.drop(columns=['diagnosis'])
    df.columns = df.columns.str.replace('_', '', regex=False)
    # Compute hash only once
    df['hash'] = med_df.apply(lambda row: hash(tuple(row)), axis=1)
    # Compute risk per unique hash
    risk_df = df.groupby('hash', as_index=False)['diagnosis'].agg(risk='mean')
    # Merge back with original data and drop hash column
    risk_df = df.merge(risk_df, on='hash').drop(columns=['hash'])
    #get desired order of columns
    with open("All_in_pytorch 2/common_ordered_columns.json", "r") as file:
        desired_order = json.load(file)["columns"]
    risk_df = risk_df[desired_order]
    return risk_df

#fix seed 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

#use Hydra to enable mutli run options and to easily change configurations
@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(conf):
    
    #set seed 
    set_seed(conf.sample_random_state) 

    #hydra params
    print(OmegaConf.to_yaml(conf))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #get data sample 
    df= pd.read_csv('Data/sparse_med_cleaned.csv') #CHANGE 
    df.drop(columns=['ID','fin_grossesse'], inplace=True)
    df_risk = get_Comb(df)
    #df_risk['hash'] = df_risk.drop(columns=['diagnosis']).apply(lambda row: hash(tuple(row)), axis=1)
    #df_risk_temp = df_risk[['hash']].drop_duplicates()
    #df_sampler_temp = df_risk_temp.sample(frac=0, random_state=conf.sample_random_state) 
    #df_sampler = df_risk[df_risk['hash'].isin(df_sampler_temp['hash'])]  # NO TRAIN
    #df_sampler_new = df_risk[~df_risk['hash'].isin(df_sampler_temp['hash'])].drop(columns=['hash']) 
    df_sampler_new = df_risk.drop_duplicates()
    print("All data shape :", df_risk.shape)
    #print("Sample shape", df_sampler.shape)  # NO TRAIN
    print("Test Sample shape", df_sampler_new.shape) 

    """NO TRAIN"""
    """
    optimize_function = partial(optimize, conf=conf, device=device, df_sampler_temp=df_sampler_temp, df_sampler=df_sampler)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimize_function, n_trials=conf.trials)

    # Extract all trials
    trials = study.trials

    # Create a DataFrame to store parameters and performance values
    data = []
    for trial in trials:
        trial_data = trial.params
        trial_data['value'] = trial.value
        data.append(trial_data)
    df = pd.DataFrame(data)
    # Sort the DataFrame by performance value
    df_sorted = df.sort_values(by='value')
    save_path = f"{output_dir}/sorted_trials.csv"
    df_sorted.to_csv(save_path, index=False)
    print("The sorted DataFrame has been saved to 'sorted_trials.csv'.")
  
    #plot optimization plot
    plot_optimization_history(study)
    save_path = f"{output_dir}/optimisation_history.png"
    plt.savefig(save_path)
    plt.clf()

    #get best params
    best_params = study.best_params
    print(best_params)
    # Open a file in write mode
    with open(f"{output_dir}/best_params.txt", 'w') as file:
        # Write the best_params to the file
        file.write(str(best_params))
    print("Best parameters have been saved to best_params.txt")
    
    
    #Train 
    df_sampler.drop(columns=['hash'],inplace=True)
    X= df_sampler.iloc[:,:-2].to_numpy(dtype="float32")  #all meds columns
    y= df_sampler['diagnosis'].to_numpy(dtype="float32")
    y_risk = df_sampler.iloc[:,-2].to_numpy(dtype="float32")  #risk column
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device) #For training
    y_risk_tensor = torch.tensor(y_risk, dtype=torch.float32).unsqueeze(1).to(device) #For Evaluation 
  
    X_new = df_sampler_new.iloc[:,:-2].to_numpy(dtype="float32")
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
    Y_new = df_sampler_new.iloc[:,-2].to_numpy(dtype="float32")
    Y_new_tensor = torch.tensor(Y_new, dtype=torch.float32).unsqueeze(1).to(device) 
  
    num_inputs, num_features = X.shape[0], X.shape[1]  # Number of features
    
    losses_train, losses_test = [], []
    best_loss_train = float('inf')
    best_loss_test = float('inf')
    best_model_state = None
    loss_fn = nn.BCELoss()
    if conf.model == 'Logistic_regression':
        model = LogisticRegression_Simple(device, num_features=num_features)
        optimizer = torch.optim.Adam(model.parameters(), eps=1e-08, lr = best_params['learning_rate'])
    if conf.model == 'Logistic_regression_penalty':
            model = LogisticRegression_Penalty(device, num_features=num_features, num_inputs=num_inputs)
            # Optimizer for weights and biases (main model parameters)
            optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if name != "penalties"], eps=1e-08, lr=best_params['learning_rate'])
            # Optimizer for penalties (separate learning rate) + different direction
            optimizer_penalty = torch.optim.Adam([model.penalties], lr=best_params['learning_rate_penalty'], maximize=True)
    if conf.model == 'Neural_network' and conf.mode==0:
            model = NeuralNetwork(device, activation=best_params['activation'], num_features=num_features, layers=best_params['layers'], hidden_dim_1=best_params['hidden_dim_1'],  hidden_dim_2=(best_params['hidden_dim_2'] if "hidden_dim_2" in best_params else None))
            optimizer = torch.optim.Adam(model.parameters(), eps=1e-08, lr = best_params['learning_rate'])
    if conf.model == 'Neural_network' and conf.mode==1:
            model = NeuralNetwork(device, activation=best_params['activation'], num_features=num_features,mode=1, layers=best_params['layers'], hidden_dim_1=best_params['hidden_dim_1'],  hidden_dim_2=(best_params['hidden_dim_2'] if "hidden_dim_2" in best_params else None))
            optimizer = torch.optim.Adam(model.parameters(), eps=1e-08, lr = best_params['learning_rate'])
            optimizer_penalty = torch.optim.Adam([model.penalties], lr=best_params['learning_rate_penalty'], maximize=True)

    model.to(device)
    num_epochs = best_params['epoch']
    num_epochs = 10000
    for epoch in range(num_epochs):
        #train
        y_pred_train = model(X_tensor)
        #compute loss
        loss = loss_fn(y_pred_train, y_tensor)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        #train 
        loss_train = loss_fn(y_pred_train, y_risk_tensor)
        losses_train.append(loss_train.item())
        #test
        y_pred_test = model(X_new_tensor)
        # Compute loss
        loss_test = loss_fn(y_pred_test, Y_new_tensor)
        losses_test.append(loss_test.item())

        # Update parameters
        optimizer.step()
        if conf.model == 'Logistic_regression_penalty':
            optimizer_penalty.step()
        if conf.model == 'Neural_network' and conf.mode == 1:
            optimizer_penalty.step()

        # save best model  
        if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_model_train = model.state_dict()     
        if loss_test < best_loss_test:
                best_loss_test = loss_test
                best_model_test = model.state_dict()    
        
        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, LOSS Train:{loss_train.item():.4f}, LOSS Test: {loss_test.item():.4f}")

    # Save results to CSV file
    with open(f"{output_dir}/train_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss_train", "Loss_test"])  # Header

        for epoch, (loss_train, loss_test) in enumerate(zip(losses_train,losses_test), 1):
            writer.writerow([epoch, loss_train, loss_test])

    torch.save(best_model_train, f"{output_dir}/best_model_train.pth")
    print(f"Best model saved with train loss: {best_loss_train}")
    torch.save(best_model_test, f"{output_dir}/best_model_test.pth")
    print(f"Best model saved with loss: {best_loss_test}")

    #plot loss 
    plt.plot(range(num_epochs), losses_train, label="Train loss")
    plt.plot(range(num_epochs), losses_test, label="Test loss")
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    plt.legend()
    plt.title('Cross Entropy Loss (with probabilites) Over Epochs')
    save_path = f"{output_dir}/losses_plot.png"
    plt.savefig(save_path)
    plt.clf()

    #plot comb seen 
    df_sampler_ = df_sampler.drop(columns=['diagnosis']).value_counts().reset_index(name='count')
    X= df_sampler_.iloc[:,:-2].to_numpy(dtype="float32")
    y_risk = df_sampler_.iloc[:,-2].to_numpy(dtype="float32")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_risk_tensor = torch.tensor(y_risk, dtype=torch.float32).unsqueeze(1).to(device) 
    """


    df_sampler_new_ = df_sampler_new.drop(columns=['diagnosis']).value_counts().reset_index(name='count')
    X_new = df_sampler_new_.iloc[:,:-2].to_numpy(dtype="float32")
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
    Y_new = df_sampler_new_.iloc[:,-2].to_numpy(dtype="float32")
    Y_new_tensor = torch.tensor(Y_new, dtype=torch.float32).unsqueeze(1).to(device) 

    """
    # Scatter plot of true vs predicted values
    if conf.model == 'Logistic_regression':
            model = LogisticRegression_Simple(device, num_features=num_features)
    if conf.model == 'Logistic_regression_penalty':
            model = LogisticRegression_Penalty(device, num_features=num_features, num_inputs=num_inputs)
    if conf.model == 'Neural_network':
            model = NeuralNetwork(device, activation=best_params['activation'], num_features=num_features, mode=conf.mode, layers=best_params['layers'], hidden_dim_1=best_params['hidden_dim_1'],  hidden_dim_2=(best_params['hidden_dim_2'] if "hidden_dim_2" in best_params else None))

    model.load_state_dict(torch.load(f"{output_dir}/best_model_train.pth"))
    model.eval()  # Set model to evaluation mode
    y_pred_probas = model(X_tensor)
    sc = plt.scatter(y_risk_tensor.cpu().detach().numpy(), y_pred_probas.cpu().detach().numpy(), c=df_sampler_['count'], cmap='plasma',
                    norm=mcolors.LogNorm(vmin=df_sampler_['count'].min(), vmax=20), 
                    s=10, label="Predictions")

    plt.colorbar(sc, label="Fréquence du point (Log Scale)")
    plt.xlabel("Risque Vrai")
    plt.ylabel("Risque prédit")
    plt.title("Graphes des combinaisons de Train")
    plt.legend()
    save_path = f"{output_dir}/seen_probabilities.png"
    plt.savefig(save_path)
    plt.clf()

    data = []
    data['risk_true'] = y_risk_tensor.cpu().detach().numpy()
    data['risk_predicted'] = y_pred_probas.cpu().detach().numpy()
    df = pd.DataFrame(data)
    save_path = f"{output_dir}/seen_probabilities.csv"
    data.to_csv(save_path, index=False)
    print("The predicted seen combination DataFrame has been saved to 'seen_probabilities.csv'.")

    """
    #TO GET MODEL
    num_inputs, num_features = X_new.shape[0], X_new.shape[1]  # Number of features
    best_params = {} #À remplir
    #plot new comb
    # Scatter plot of true vs predicted values
    if conf.model == 'Logistic_regression':
            model = LogisticRegression_Simple(device, num_features=num_features)
    if conf.model == 'Logistic_regression_penalty':
            model = LogisticRegression_Penalty(device, num_features=num_features,num_inputs=num_inputs)
    if conf.model == 'Neural_network':
            model = NeuralNetwork(device, activation=best_params['activation'], num_features=num_features, mode=conf.mode, layers=best_params['layers'], hidden_dim_1=best_params['hidden_dim_1'],  hidden_dim_2=(best_params['hidden_dim_2'] if "hidden_dim_2" in best_params else None))

    model.load_state_dict(torch.load(f"{output_dir}/best_model_test.pth"))
    model.eval()  # Set model to evaluation mode
    y_pred_probas = model(X_new_tensor)
    sc = plt.scatter(Y_new_tensor.cpu().detach().numpy(), y_pred_probas.cpu().detach().numpy(), c=df_sampler_new_['count'], cmap='plasma',
                    norm=mcolors.LogNorm(vmin=df_sampler_new_['count'].min(), vmax=20), 
                    s=10, label="Predictions")

    plt.colorbar(sc, label="Fréquence du point (Log Scale)")
    plt.xlabel("Risque Vrai")
    plt.ylabel("Risque prédit")
    plt.title("Graphes des combinaisons de Test")
    plt.legend()
    save_path = f"{output_dir}/new_probabilities.png"
    plt.savefig(save_path)
    plt.clf()

    data = pd.DataFrame()
    data['risk_true'] = Y_new_tensor.cpu().detach().numpy().flatten()
    data['risk_predicted'] = y_pred_probas.cpu().detach().numpy().flatten()
    df = pd.DataFrame(data)
    save_path = f"{output_dir}/new_probabilities.csv"
    data.to_csv(save_path, index=False)
    print("The predicted new combination DataFrame has been saved to 'new_probabilities.csv'.")





if __name__ == "__main__":
    my_app()





