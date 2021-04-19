
""" 
Model definitions
"""

import torch 
import torch.nn as nn 
import common
import train_utils
import networks


class AbstractGeneration:

    def __init__(self, args):
        self.args = args 
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Model, optimizer, scheduler 
        self.model = networks.get_gpt2_model(freeze_layers=self.config["model"]["freeze_layers"])
        self.optim = train_utils.get_optimizer(config=self.config["optimizer"], params=self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config={**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)
        
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        