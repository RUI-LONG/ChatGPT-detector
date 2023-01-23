import torch
import streamlit as st
from transformers import BertTokenizerFast, GPT2LMHeadModel

class Perplexity:
    def load_model():
        # code to load the model
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        model = GPT2LMHeadModel.from_pretrained("ckiplab/gpt2-base-chinese")
        return tokenizer, model
        
    def set_model(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)

    def calculate(self, text):
        stride = 512
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.n_positions
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        ppl = torch.exp(torch.stack(nlls).sum() / len(text)).item()
        return int(ppl)

def classify_perplexity(score):
    if score <= 18:
        return "AI"
    if score >= 30:
        return "Human"
    return "Unkown"
