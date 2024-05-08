import flwr as fl
import torch
from collections import OrderedDict
from torch.optim import SGD

from model import CNN, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 train_loader,
                 val_loader,
                 num_classes):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CNN(num_classes)
        
    def set_params(self, params):
        # Update parameters of the model
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def get_params(self):
        # Return the current parameters of the model
        return [val.cpu().numpy() for _, val in self.model.state_dict.items()]
    
    def fit(self, params, cfg):
        # Update parameters from the sever
        self.set_params(params)
        
        lr = cfg["lr"]
        momentum = cfg["momentum"]
        epochs = cfg["local_epochs"]
        optim = SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        # Local training
        train(self.model, self.train_loader, optim, epochs, self.device)
        
        return self.get_params(), len(self.train_loader), {}  # len of loader is for FedAVG, dict is for additional info sent to server
    
    def evaluate(self, params):
        # Evaluate on the parameters sent by the server
        self.set_params(params)
        
        loss, accuracy = test(self.model, self.val_loader, self.device)
        
        return float(loss), len(self.val_loader), {"accuracy": accuracy}