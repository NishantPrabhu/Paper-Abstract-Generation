
""" 
Model definitions
"""

import torch 
import wandb
import common
import losses
import metrics
import networks
import data_utils
import train_utils
import torch.nn as nn 
from transformers import BertModel
from transformers import BertTokenizer 


class AbstractGeneration:

    def __init__(self, args):
        self.args = args 
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args)

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader = data_utils.get_dataloaders(
            self.config["data"]["root"], self.config["data"]["val_split"], self.config["data"]["batch_size"])

        # Model, optimizer, scheduler 
        self.encoder = BertModel.from_pretrained(self.config["encoder"]["name"])
        self.encoder.resize_token_embeddings(len(self.train_loader.tokenizer))
        self.decoder = networks.Decoder(self.config["decoder"], self.encoder.embeddings, self.train_loader.tokenizer.vocab_size)

        self.optim = train_utils.get_optimizer(
            config=self.config["optimizer"], params=list(self.encoder.parameters())+list(self.decoder.parameters()))
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

    def get_metrics(self):
        all_abstracts, all_outputs = [], []
        for idx in range(self.val_loader):
            batch = self.val_loader.flow()
            _, title_tokens, abstracts, abstract_tokens = batch 
            with torch.no_grad():
                encoder_out = self.encoder(title_tokens)
                out_sentences = self.generate_abstracts(encoder_out)
            all_abstracts.extend(abstracts)
            all_outputs.extend(out_sentences)

        bleu_scores = metrics.bleu_score(all_outputs, all_abstracts)
        avg_length = metrics.average_text_lengths(all_outputs)
        return {**bleu_scores, **avg_length}

    def generate_abstract(self, encoder_outs):
        idx2word = {idx: word for word, idx in dict(self.test_loader.tokenizer.vocab).items()}
        output_sentences = []

        for i in range(encoder_outs.size(0)):
            generated_text = "[BOS] "
            while len(generated_text.split()) < self.maxlen:
                input_ids = self.test_loader.tokenizer(generated_text)["input_ids"]
                with torch.no_grad():
                    word_probs = self.decoder(input_ids, encoder_outs[i])
                word = idx2word.get(word_probs.argmax(dim=1), '')
                generated_text += word + " "
            sent = " ".join([w for w in generated_text.split() if w not in ["[BOS], [EOS], [PAD]"]])   
            output_sentences.append(sent)
        return output_sentences

    def train_one_step(self, batch):
        _, title_tokens, abstracts, abstract_tokens = batch 
        abstract_inp, abstract_trg = abstract_tokens[:, :-1], abstract_tokens[:, 1:]
        title_enc = self.encoder(title_tokens)
        abstract_out = self.decoder(abstract_inp, encoder_out=title_enc)
        loss = self.criterion(abstract_out, abstract_trg)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    def validate_one_step(self, batch):
        _, title_tokens, abstracts, abstract_tokens = batch
        abstract_inp, abstract_trg = abstract_tokens[:, :-1], abstract_tokens[:, 1:]
        title_enc = self.encoder(title_tokens)
        with torch.no_grad():
            abstract_out = self.decoder(abstract_inp, encoder_out=title_enc)
        
        loss = self.criterion(abstract_out, abstract_trg)
        return {"loss": loss.item()}

    def get_test_predictions(self):
        if self.args["load"] is None:
            self.load_model(self.output_dir)
            
        test_ids, test_titles, test_preds = [], [], []
        for idx, batch in enumerate(self.test_loader):
            titles, title_tokens, ids = batch
            with torch.no_grad():
                encoder_out = self.encoder(title_tokens)
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
            self.model.train()
            train_meter = common.AverageMeter()

            for idx, batch in enumerate(self.train_loader):
                train_metrics = self.train_one_step(batch)
                train_meter.add(train_metrics)
                wandb.log({"Train loss": train_meter.return_metrics()["loss"]})
                common.progress_bar(progress=(idx+1)/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1.0, status=train_meter.return_msg())
            self.adjust_learning_rate(epoch+1)

            if (epoch+1) % self.config["eval_every"] == 0:
                self.logger.record(f'Epoch {epoch+1}/{self.config["epochs"]}', mode='val')
                self.model.eval()
                val_meter = common.AverageMeter()
                
                for idx, batch in enumerate(self.val_loader):
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=(idx+1)/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1.0, status=val_meter.return_msg())
                metrics = self.get_metrics()
                wandb.log({
                    "Validation loss": val_meter.return_metrics()["loss"],
                    "BLEU-1": metrics["bleu_1"], "BLEU-2": metrics["bleu_2"], "BLEU-3": metrics["bleu_3"],
                    "Average length": metrics["avg_length"], "Epoch": epoch+1
                })

                if val_meter.return_metrics()["loss"] < self.best_val_loss:
                    self.best_val_loss = val_meter.return_metrics()["loss"]
                    self.save_model()

        print(f"\n\n{common.COLORS['yellow']}[INFO] Finished training! Generating test predictions...{common.COLORS['end']}")
        self.get_test_predictions()