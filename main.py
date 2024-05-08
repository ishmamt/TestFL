import hydra
from omegaconf import OmegaConf, DictConfig
import flwr as fl

from dataset import load_dataset
from client import generate_client_function

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Parsse config
    print(OmegaConf.to_yaml(cfg))
    
    # Prepare Dataset
    train_loaders, val_loaders, test_loader = load_dataset(cfg.num_clients, cfg.batch_size)
    
    # Define Clients
    client_function = generate_client_function(train_loaders, val_loaders, cfg.num_classes)
    
    # Define Strategy
    strategy = fl.server.strategy.FedAVG(fraction_fit=)
    
    
if __name__ == "__main__":
    main()