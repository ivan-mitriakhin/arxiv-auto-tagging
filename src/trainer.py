import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from torcheval.metrics.functional import multilabel_accuracy

from model import MultilabelModel

class Trainer():

    def __init__(self, config, save_epoch=True):
        self.config = config
        self.n_epochs = config['n_epochs']
        self.optimizer = None
        self.opt_fn = lambda model: AdamW(model.parameters(), config['lr'])
        self.model = None
        self.loss_fn = BCEWithLogitsLoss()
        self.device = config['device']
        self.verbose = config.get('verbose', True)
        self.save_epoch = save_epoch

    def fit(self, model, train_dataloader, val_dataloader):
        self.model = model.to(self.device)
        self.optimizer = self.opt_fn(model)
            

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            self.train_epoch(train_dataloader)
            self.val_epoch(val_dataloader)

            if self.save_epoch:
                self.save(f"/model/b-deberta-v3_{epoch + 1}.ckpt")
        
        return self.model.eval()

    def train_epoch(self, train_dataloader):
        self.model.train()
        
        if self.verbose:
            train_dataloader = tqdm(train_dataloader)
        
        for batch in train_dataloader:
            ids = batch['ids'].to(self.device, dtype=torch.long)
            mask = batch['mask'].to(self.device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
            targets = batch['targets'].to(self.device, dtype=torch.long)

            outputs = self.model(ids, mask, token_type_ids)
            loss = self.loss_fn(outputs, targets.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def val_epoch(self, val_dataloader):
        self.model.eval()
        all_logits = []
        all_labels = []
        if self.verbose:
            val_dataloader = tqdm(val_dataloader)

        with torch.no_grad():
            for batch in val_dataloader:
                ids = batch['ids'].to(self.device, dtype=torch.long)
                mask = batch['mask'].to(self.device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                targets = batch['targets'].to(self.device, dtype=torch.long)

                outputs = self.model(ids, mask, token_type_ids)
                all_logits.extend(outputs)
                all_labels.extend(targets)
        
        all_logits = torch.stack(all_logits, dim=0).to(self.device)
        all_labels = torch.stack(all_labels, dim=0).to(self.device)
        loss = self.loss_fn(all_logits, all_labels.float()).item()

        pred = torch.sigmoid(all_logits).cpu().detach().numpy()
        pred = (pred >= 0.5).astype(int)
        true = all_labels.cpu().detach().numpy()

        acc = multilabel_accuracy(torch.from_numpy(pred), torch.from_numpy(true), criteria='hamming')
        f1 = f1_score(true, pred, average='samples')

        print(f"Loss: {loss}")
        print(f"Accuracy: {acc}")
        print(f"F1: {f1}")

    def predict(self, test_dataloader):
        if not self.model:
            raise RuntimeError("You should train the model first.")
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in test_dataloader:
                ids = batch['ids'].to(self.device, dtype=torch.long)
                mask = batch['mask'].to(self.device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask, token_type_ids)
                pred = torch.sigmoid(outputs).cpu().detach().numpy()
                pred = (pred >= 0.5).astype(int)
                preds.extend(pred)
        
        return preds
    

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        checkpoint = {
            "config": self.model.config,
            "trainer_config": self.config,
            "model_name": self.model.model_name,
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path)
        keys = ["config", "trainer_config", "model_state_dict"]
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")
        new_model = MultilabelModel(
            ckpt['model_name'],
            ckpt["config"]
        )
        new_model.load_state_dict(ckpt["model_state_dict"])
        new_trainer = cls(ckpt["trainer_config"])
        new_trainer.model = new_model
        new_trainer.model.to(new_trainer.device)
        return new_trainer
        
