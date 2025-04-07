import torch
import torch.nn as nn

class LogisticRegression_Simple(torch.nn.Module):
    def __init__(self, device, num_features):
        super(LogisticRegression_Simple, self).__init__()
        self.device = device
        self.linear = torch.nn.Linear(num_features,1).to(device)

        # Initialize weights
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        
        # Initialize biases
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, X):
        logits = self.linear(X)
        probas = torch.sigmoid(logits)
        return probas

  
class LogisticRegression_Penalty(torch.nn.Module):
    def __init__(self, device, num_features, num_inputs):
        super(LogisticRegression_Penalty, self).__init__()
        self.device = device
        self.linear= torch.nn.Linear(num_features,1).to(device)

        # Initialize weights
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        
        # Initialize biases
        torch.nn.init.zeros_(self.linear.bias)
        
        # Penalty term
        self.penalties = torch.ones((num_inputs, 1), requires_grad=True, dtype=torch.float32, device = device)


    def forward(self, X):
        logits = self.linear(X)
        probas = torch.sigmoid(logits)
        return probas
    
    def compute_loss(self, y_pred, y_true):
        loss = - torch.mean(self.penalties * y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        return loss 
    

# mode = 0 means BCE loss function / mode = 1 means modified loss function 
class NeuralNetwork(torch.nn.Module):
    def __init__(self, device, activation, num_features, mode=0, layers=1, hidden_dim_1=200, hidden_dim_2=200):
        super(NeuralNetwork, self).__init__()
        self.mode = mode
        self.device = device
        self.layers = layers
        self.activation = activation
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        #initialize parameters with conditions 
        if self.layers == 1 :
            self.linear1 = torch.nn.Linear(num_features,self.hidden_dim_1).to(device)
            self.linear2 = torch.nn.Linear(self.hidden_dim_1, 1).to(device)
            # Initialize weights
            torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
            # Initialize biases
            torch.nn.init.zeros_(self.linear1.bias)
            torch.nn.init.zeros_(self.linear2.bias)
            if self.mode == 1: 
                self.penalties = torch.ones((hidden_dim_1, 1), requires_grad=True, dtype=torch.float32, device = device)

        if self.layers == 2 :
            self.linear1 = torch.nn.Linear(num_features,self.hidden_dim_1).to(device)
            self.linear2 = torch.nn.Linear(self.hidden_dim_1, self.hidden_dim_2).to(device)
            self.linear3 = torch.nn.Linear(self.hidden_dim_2, 1).to(device)
            # Initialize weights
            torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
            # Initialize biases
            torch.nn.init.zeros_(self.linear1.bias)
            torch.nn.init.zeros_(self.linear2.bias)
            torch.nn.init.zeros_(self.linear3.bias)
            if self.mode == 1: 
                self.penalties = torch.ones((hidden_dim_2, 1), requires_grad=True, dtype=torch.float32, device = device)
        

    def forward(self, X):

        if self.activation == 'LRELU':
            m = nn.LeakyReLU()
        if self.activation == 'RELU':
            m = nn.ReLU()

        if self.layers == 1:
            layer1_out = self.linear1(X)
            logits = self.linear2(m(layer1_out))
        
        if self.layers == 2: 
            layer1_out = self.linear1(X)
            layer2_out = self.linear2(m(layer1_out))
            logits = self.linear3(m(layer2_out))
            
        probas = torch.sigmoid(logits) 
        return probas      
    
    def compute_loss(self, y_pred, y_true):
        loss = - torch.mean(self.penalties * y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        return loss 