
""" 
Model definitions
"""

import os 
import torch 
import wandb
import common
import metrics
import networks
import data_utils
import train_utils
import numpy as np
import pandas as pd 
import torch.nn as nn 
from transformers import BartForConditionalGeneration


class AbstractGeneration:

    def __init__(self, args):
        self.args = args 
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Model, optimizer, scheduler 
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(self.device)
        self.optim = train_utils.get_optimizer(config=self.config["optimizer"], params=self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config={**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)
        
        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader = data_utils.get_dataloaders(
            self.config["data"]["root"], self.config["data"]["val_split"], self.config["data"]["batch_size"], self.device)
        
        # Logging and metrics
        run = wandb.init(project="abstract-generation-dl-hack")
        self.logger.write(f"Wandb: {run.get_url()}", mode='info')
        self.best_val_loss = np.inf

        # Load model if specified
        if args["load"] is not None:
            self.load_model(args["load"])

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.ckpt"))

    def load_model(self, path):
        if os.path.exists(os.path.join(path, "best_model.ckpt")):
            state = torch.load(os.path.join(path, "best_model.ckpt"))
            self.model.load_state_dict(state)
            self.logger.print(f"Successfully loaded model from {path}", mode='info')
        else:
            raise NotImplementedError(f"No saved model found in {path}")

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()

    def generate_abstracts(self, title_tokens):
        out_tokens = self.model.generate(
            title_tokens, 
            do_sample = True,
            max_length = self.config["test"]["max_length"],
            top_p = self.config["test"].get("top_p", 0.95),
            top_k = self.config["test"].get("top_k", 50),
            no_repeat_ngram_size = self.config["test"].get("no_repeat_ngram_size", 2)
        )
        out_sentences = []
        for i in range(out_tokens.size(0)):
            out_sentences.append(self.test_loader.tokenizer.decode(out_tokens[i], skip_special_tokens=True))
        return out_sentences

    def train_one_step(self, batch):
        self.model.train()
        _, title_tokens, abstracts, abstract_tokens = batch 
        output = self.model(input_ids=title_tokens, labels=abstract_tokens, return_dict=True)
        loss = output["loss"]

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        torch.cuda.empty_cache()
        acc = metrics.accuracy(output, abstract_tokens)
        return {"loss": loss.item(), **acc}

    def validate_one_step(self, batch):
        self.model.eval()
        _, title_tokens, abstracts, abstract_tokens = batch 
        with torch.no_grad():
            output = self.model(input_ids=title_tokens, labels=abstract_tokens, return_dict=True)
        loss = output["loss"]
        acc = metrics.accuracy(output, abstract_tokens)
        return {"loss": loss.item(), **acc}

    def get_test_predictions(self):
        if self.args["load"] is None:
            self.load_model(self.output_dir)
            
        test_ids, test_titles, test_preds = [], [], []
        for idx in range(len(self.test_loader)):
            batch = self.test_loader.flow()
            titles, title_tokens, ids = batch
            with torch.no_grad():
                out_sentences = self.generate_abstracts(title_tokens)

            test_ids.extend(ids)
            test_preds.extend(out_sentences)
            test_titles.extend(titles)
            common.progress_bar(progress=(idx+1)/len(self.test_loader), status="")
            
        common.progress_bar(progress=1.0, status="")
        sub_df = pd.DataFrame({"id": test_ids, "title": test_titles, "abstract": test_preds})
        sub_df["id"] = sub_df["id"].astype("int")
        sub_df = sub_df.sort_values(by="id", ascending=True)
        sub_df.to_csv(os.path.join(self.output_dir, "test_predictions.csv"), index=False)

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.logger.record(f'Epoch {epoch+1}/{self.config["epochs"]}', mode='train')
            self.model.train()
            train_meter = common.AverageMeter()

            for idx in range(len(self.train_loader)):
                batch = self.train_loader.flow()
                train_metrics = self.train_one_step(batch)
                train_meter.add(train_metrics)
                wandb.log({"Train loss": train_meter.return_metrics()["loss"]})
                common.progress_bar(progress=(idx+1)/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1.0, status=train_meter.return_msg())
            wandb.log({"Train accuracy": train_meter.return_metrics()["accuracy"], "Epoch": epoch+1})
            self.adjust_learning_rate(epoch+1)

            if (epoch+1) % self.config["eval_every"] == 0:
                self.logger.record(f'Epoch {epoch+1}/{self.config["epochs"]}', mode='val')
                self.model.eval()
                val_meter = common.AverageMeter()
                
                for idx in range(len(self.val_loader)):
                    batch = self.val_loader.flow()
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=(idx+1)/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1.0, status=val_meter.return_msg())
                wandb.log({
                    "Validation loss": val_meter.return_metrics()["loss"],
                    "Validation accuracy": val_meter.return_metrics()["accuracy"],
                    "Epoch": epoch+1
                })

                if val_meter.return_metrics()["loss"] < self.best_val_loss:
                    self.best_val_loss = val_meter.return_metrics()["loss"]
                    self.save_model()

        print(f"\n\n{common.COLORS['yellow']}[INFO] Finished training! Generating test predictions...{common.COLORS['end']}")
        self.get_test_predictions()