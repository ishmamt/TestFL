from omegaconf import DictConfig

def get_on_fit_config_function(cfg:DictConfig):
    # Provides a on fit config function which the server can evoke when trying to fit on the client
    def on_fit_config_function(server_round):
        # Provides server_round for customized on fit behavior
        
        return {"lr": cfg.lr, 
                "momentum": cfg.momentum, 
                "local_epoch": cfg.local_epoch}
    
    return on_fit_config_function


def get_on_eval_config_function(cfg:DictConfig):
    # Provides a on fit config function which the server can evoke when trying to evaluate on the client
    def on_eval_config_function(server_round, params):
        # Provides server_round for customized on fit behavior
        
        return {"lr": cfg.lr, 
                "momentum": cfg.momentum, 
                "local_epoch": cfg.local_epoch}
    
    return on_eval_config_function