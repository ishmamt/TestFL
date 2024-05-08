import hydra
from omegaconf import OmegaConf, DictConfig

from dataset import load_dataset

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Parsse config
    print(OmegaConf.to_yaml(cfg))
    
    # Prepare Dataset
    train_loaders, val_loaders, test_loader = load_dataset(cfg.num_clients, cfg.batch_size)
    print(len(train_loaders), len(train_loaders[0].dataset))
    
    # Define Clients
    


if __name__ == "__main__":
    main()