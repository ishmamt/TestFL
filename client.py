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
        
        self.model = CNN(num_classes).to(self.device)
        
    def set_params(self, params):
        # Update parameters of the model
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, params, cfg):
        # Update parameters from the sever
        self.set_params(params)
        
        lr = cfg["lr"]
        momentum = cfg["momentum"]
        epochs = cfg["local_epochs"]
        optim = SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        # Local training
        train(self.model, self.train_loader, optim, epochs, self.device)
        
        