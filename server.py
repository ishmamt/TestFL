from omegaconf import DictConfig
import torch
from collections import OrderedDict

from model import CNN, test

def get_on_fit_config_function(cfg:DictConfig):
    # Provides a on fit config function which the server can evoke when trying to fit on the client
    def on_fit_config_function(server_round):
        # Provides server_round for customized on fit behavior
        
        return {"lr": cfg.lr, 
                "momentum": cfg.momentum, 
                "local_epochs": cfg.local_epochs}
    
    return on_fit_config_function


def get_eval_function(num_classes, test_loader):
    # Provides a on fit config function which the server can evoke when trying to evaluate on the client
    def eval_function(server_round, params, cfg):
        # Provides server_round for customized on eval behavior
        model = CNN(num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(model, test_loader, device)
        
        return loss, {"accuracy": accuracy}
    
    return eval_function