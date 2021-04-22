
""" 
Model definitions
"""

import os
import torch 
import wandb
import common
import losses
import metrics
import networks
import data_utils
import train_utils
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer 


class AbstractGeneration:

    def __init__(self, args):
        self.args = args 
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader = data_utils.get_dataloaders(
            self.config["data"]["root"], self.config["data"]["val_split"], self.config["data"]["batch_size"], self.device)

        # Model, optimizer, scheduler 
        self.encoder = BertModel.from_pretrained(self.config["encoder"]["name"]).to(self.device)
        self.encoder.resize_token_embeddings(len(self.train_loader.tokenizer))
        self.decoder = networks.Decoder(
            self.config["decoder"], self.encoder.embeddings, self.train_loader.tokenizer.vocab_size).to(self.device)

        # Setting encoder untrainable
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.optim = train_utils.get_optimizer(
            config=self.config["optimizer"], params=self.decoder.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config={**self.config["scheduler"], "epochs": self.config["epochs"]}, optimizer=self.optim)

        if self.warmup_epochs > 0:
            self.warmup_rate = (self.config["optimizer"]["lr"] - 1e-12) / self.warmup_epochs
        
        # Logging, criterion and metrics
        run = wandb.init(project="abstract-generation-dl-hack")
        self.logger.write(f"Wandb: {run.get_url()}", mode='info')
        self.criterion = losses.GenerationLoss()
        self.best_val_loss = np.inf

        # Load model if specified
        if args["load"] is not None:
            self.load_model(args["load"])

    def trainable(self, val):
        if val:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def save_model(self):
        state = {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}
        torch.save(state, os.path.join(self.output_dir, "best_model.ckpt"))

    def load_model(self, path):
        if os.path.exists(os.path.join(path, "best_model.ckpt")):
            state = torch.load(os.path.join(path, "best_model.ckpt"))
            self.encoder.load_state_dict(state["encoder"])
            self.decoder.load_state_dict(state["decoder"])
            self.logger.print(f"Successfully loaded model from {path}", mode='info')
        else:
            raise NotImplementedError(f"No saved model found in {path}")

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()

    def get_metrics(self, epoch):
        all_abstracts, all_outputs = [], []
        self.logger.record(f"Epoch {epoch}/{self.config['epochs']}: Computing metrics", mode='val')
        
        for idx in range(len(self.val_loader)):
            batch = self.val_loader.flow()
            _, title_tokens, abstracts, abstract_tokens = batch 
            with torch.no_grad():
                encoder_out = self.encoder(title_tokens)[0]
                out_sentences = self.generate_abstracts(encoder_out)
            all_abstracts.extend(abstracts)
            all_outputs.extend(out_sentences)
            break

        bleu_scores = metrics.bleu_score(all_outputs, all_abstracts)
        avg_length = metrics.average_text_lengths(all_outputs)
        eval_metrics = {**bleu_scores, **avg_length} 
        status = " ".join(["[{}] {:.4f}".format(k, v) for k, v in eval_metrics.items()])

        common.progress_bar(progress=1.0, status=status)
        return eval_metrics, status

    def generate_abstracts(self, encoder_outs):
        idx2word = {idx: word for word, idx in dict(self.test_loader.tokenizer.vocab).items()}
        output_sentences = []

        for i in range(encoder_outs.size(0)):
            generated_text = ""
            count = 0
            while (count < self.decoder.maxlen) and (not generated_text.endswith("[SEP]")):
                input_ids = self.test_loader.tokenizer(
                    generated_text, truncation=True, return_tensors="pt")["input_ids"].to(self.device)
                with torch.no_grad():
                    word_probs = F.softmax(self.decoder(input_ids[:, :-1], encoder_outs[i].unsqueeze(0)), dim=-1)[:, 0, :]
                    word_probs = word_probs.squeeze(1).detach().cpu()
                word = idx2word[int(word_probs.argmax(dim=1).item())]
                generated_text += word + " "
                count += 1

            common.progress_bar(progress=(i+1)/encoder_outs.size(0), status="")
            sent = " ".join([w for w in generated_text.split() if w not in ["[SEP], [CLS], [PAD]"]])   
            output_sentences.append(sent)
        return output_sentences

    def train_one_step(self, batch):
        _, title_tokens, abstracts, abstract_tokens = batch 
        abstract_inp, abstract_trg = abstract_tokens[:, :-1], abstract_tokens[:, 1:]
        
        title_enc = self.encoder(title_tokens)[0]
        abstract_out = self.decoder(abstract_inp, encoder_out=title_enc)
        loss, acc = self.criterion(abstract_out, abstract_trg)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item(), "accuracy": acc}

    def validate_one_step(self, batch):
        _, title_tokens, abstracts, abstract_tokens = batch
        abstract_inp, abstract_trg = abstract_tokens[:, :-1], abstract_tokens[:, 1:]
        
        with torch.no_grad():
            title_enc = self.encoder(title_tokens)[0]
            abstract_out = self.decoder(abstract_inp, encoder_out=title_enc)
        
        loss, acc = self.criterion(abstract_out, abstract_trg)
        return {"loss": loss.item(), "accuracy": acc}

    def get_test_predictions(self):
        if self.args["load"] is None:
            self.load_model(self.output_dir)
            
        test_ids, test_titles, test_preds = [], [], []
        for idx in range(len(self.test_loader)):
            batch = self.test_loader.flow()
            titles, title_tokens, ids = batch
            with torch.no_grad():
                encoder_out = self.encoder(title_tokens)[0]
                out_sentences = self.generate_abstracts(encoder_out)

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
            train_meter = common.AverageMeter()
            self.trainable(True)

            for idx in range(len(self.train_loader)):
                batch = self.train_loader.flow()
                train_metrics = self.train_one_step(batch)
                train_meter.add(train_metrics)
                wandb.log({"Train loss": train_meter.return_metrics()["loss"]})
                common.progress_bar(progress=(idx+1)/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1.0, status=train_meter.return_msg())
            wandb.log({"Train accuracy": train_meter.return_metrics()["accuracy"], "Epoch": epoch+1})
            self.logger.write(train_meter.return_msg(), mode='train')
            self.adjust_learning_rate(epoch+1)

            if (epoch+1) % self.config["eval_every"] == 0:
                self.logger.record(f'Epoch {epoch+1}/{self.config["epochs"]}', mode='val')
                val_meter = common.AverageMeter()
                self.trainable(False)
                
                for idx in range(len(self.val_loader)):
                    batch = self.val_loader.flow()
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=(idx+1)/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1.0, status=val_meter.return_msg())
                eval_metrics, status_msg = self.get_metrics(epoch+1)
                self.logger.write(val_meter.return_msg() + status_msg, mode='val')
                wandb.log({
                    "Validation loss": val_meter.return_metrics()["loss"],
                    "Validation accuracy": val_meter.return_metrics()["accuracy"],
                    "BLEU-1": eval_metrics["bleu_1"], "BLEU-2": eval_metrics["bleu_2"], "BLEU-3": eval_metrics["bleu_3"],
                    "Average length": eval_metrics["avg length"], "Epoch": epoch+1
                })

                if val_meter.return_metrics()["loss"] < self.best_val_loss:
                    self.best_val_loss = val_meter.return_metrics()["loss"]
                    self.save_model()

        print(f"\n\n{common.COLORS['yellow']}[INFO] Finished training! Generating test predictions...{common.COLORS['end']}")
        self.get_test_predictions()