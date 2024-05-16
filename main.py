import hydra
from omegaconf import OmegaConf, DictConfig
import flwr as fl

import torch
from collections import OrderedDict

from dataset import load_dataset
from client import generate_client_function
from server import get_on_fit_config_function, get_eval_function

from model import CNN

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Parsse config
    print(OmegaConf.to_yaml(cfg))
    
    # Prepare Dataset
    train_loaders, val_loaders, test_loader = load_dataset(cfg.num_clients, cfg.batch_size)
    
    # Define Clients
    client_function = generate_client_function(train_loaders, val_loaders, cfg.num_classes)
    
    # Define Strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.0001, 
                                         min_fit_clients=cfg.num_clients_per_round_fit, 
                                         fraction_evaluate=0.0001, 
                                         min_evaluate_clients=cfg.num_clients_per_round_eval, 
                                         min_available_clients=cfg.num_clients, 
                                         on_fit_config_fn=get_on_fit_config_function(cfg.config_fit),
                                         evaluate_fn=get_eval_function(cfg.num_classes, test_loader))
    
    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_function,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy
    )
    
    
if __name__ == "__main__":
    main()