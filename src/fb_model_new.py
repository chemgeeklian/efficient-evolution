import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO

class FBModel(object):
    def __init__(self, model_path, device='cuda:0'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForMaskedLM.from_pretrained(model_path).to(device)
        self.device = device
        self.model.eval()

    def predict_sequence_prob(self, seq):
        input_ids = self.tokenizer(seq, return_tensors="pt")['input_ids'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            logits = outputs.logits
            attention = outputs.attentions
        return logits, attention