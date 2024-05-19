import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from dataset import AbstractsDataset
from model import MultilabelModel
from trainer import Trainer
import joblib

MODEL_PATH = "../model/"
MAX_SEQ_LEN = 350

def flatten(xss):
    return [x for xs in xss for x in xs]

def make_prediction(title: str, abstract: str):
    t = Trainer.load(MODEL_PATH + "b-deberta-v3_14.ckpt")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "tokenizer")

    test = pd.DataFrame({"titles": [title], "abstracts": [abstract]})

    test_dataset = AbstractsDataset(test, tokenizer, MAX_SEQ_LEN)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    mlb = joblib.load('../utils/multilabelbinarizer.pkl')

    preds = t.predict(test_dataloader)
    preds  = np.array(preds)
    preds = mlb.inverse_transform(preds)
    preds = flatten(preds)
    
    return preds