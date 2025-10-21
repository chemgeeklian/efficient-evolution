import torch
from transformers import EsmForMaskedLM, EsmTokenizer
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FBModel(object):
    def __init__(self, model_path, device='cuda:0'):
        self.tokenizer = EsmTokenizer.from_pretrained(model_path)
        self.model = EsmForMaskedLM.from_pretrained(model_path).to(device)
        self.device = device
        self.model.eval()

    def predict_sequence_prob(self, seq):
        """
        Predict logits for one or many sequences.
        Accepts a single string or a list of strings. Returns a numpy array of
        shape (batch, seq_len, vocab) when given a batch, or (seq_len, vocab)
        for a single input string.
        """
        single = False
        if isinstance(seq, str):
            seqs = [seq]
            single = True
        else:
            seqs = list(seq)

        inputs = self.tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, return_dict=True)
            logits = outputs.logits  # (batch, seq_len, vocab)

        logits = logits.cpu().numpy()
        return logits[0] if single else logits

    def encode(self, seq):
        """
        Encode one or many sequences and return per-residue embeddings.
        Input: single string or list of strings.
        Output: if single string, returns np.ndarray (L, D). If list, returns
        a list of np.ndarray where each element is (Li, D) for sequence i.
        """
        single = False
        if isinstance(seq, str):
            seqs = [seq]
            single = True
        else:
            seqs = list(seq)

        # Tokenize batch with padding; set a conservative max_length
        inputs = self.tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, dim)
        attention_mask = inputs.get('attention_mask')  # (batch, seq_len)

        last_hidden = last_hidden.cpu()
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()

        embeddings = []
        batch_size = last_hidden.shape[0]
        for i in range(batch_size):
            seq_len = attention_mask.shape[1] if attention_mask is not None else last_hidden.shape[1]
            if attention_mask is not None:
                mask = attention_mask[i].bool()
                # select only real tokens (non-padded)
                emb = last_hidden[i, mask, :].numpy()
            else:
                emb = last_hidden[i, :seq_len, :].numpy()
            embeddings.append(emb)

        return embeddings[0] if single else embeddings

    def decode(self, embedding):
        """
        Decode per-residue embeddings back to logits over the vocabulary.
        Accepts a single (L, D) numpy array or a list of such arrays. Returns a
        numpy array (L, V) for single input or a list of arrays for a batch.
        """
        single = False
        if isinstance(embedding, np.ndarray):
            embs = [embedding]
            single = True
        else:
            embs = list(embedding)

        outputs = []
        with torch.no_grad():
            for emb in embs:
                t = torch.from_numpy(emb).to(self.device)
                # lm_head supports inputs of shape (..., hidden_dim)
                logits = self.model.lm_head(t)
                outputs.append(logits.cpu().numpy())

        return outputs[0] if single else outputs

    
if __name__ == '__main__':
    import torch
    import numpy as np

    # ====== Basic setup ======
    model_path = '/eagle/projects/FoundEpidem/xlian/checkpoints/facebook_esm2_t33_650M_UR50D'
    #model_path = 'facebook/esm2_t30_150M_UR50D'
    #model_path = '/eagle/projects/FoundEpidem/xlian/checkpoints/dgoa-esm/dgoa-runs-150M/checkpoint-2210'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"[INFO] Loading ESM model from: {model_path}")
    fb = FBModel(model_path, device=device)

    # ====== Test sequences ======
    test_seqs = ["MQWQTKLPLIAILRGITPDEALAHVGAVIDAGFDAVEIPLNSPQWEQSIPAIVDAYGDKALIGAGTVLKPEQVDALARMGCQLIVTPNIHSEVIRRAVGYGMTVCPGCATATEAFTALEAGAQALKIFPSSAFGPQYIKALKAVLPSDIAVFAVGGVTPENLAQWIDAGCAGAGLGSDLYRAGQSVERTAQQAAAFVKAYREAVQ","MQWQTNLPLIAILRGITPDEALAHVGAVIDAGFDAVEIPLNSPQWEKSIPQVVDAYGEQALIGAGTVLQPEQVDRLAAMGCRLIVTPNIQPEVIRRAVGYGMTVCPGCATASEAFSALDAGAQALKIFPSSAFGPDYIKALKAVLPPEVPVFAVGGVTPENLAQWINAGCVGAGLGSDLYRAGQSVERTAQQAAAFVKAYREAVK"]
    print(f"[INFO] Testing encode/decode on {len(test_seqs)} sequences")

    # ====== Encode ======
    embeddings = fb.encode(test_seqs)
    if isinstance(embeddings, np.ndarray):
        embeddings = [embeddings]  # make it list for unified handling

    for i, emb in enumerate(embeddings):
        print(f"[RESULT] Seq {i} embedding shape: {emb.shape}  (L={emb.shape[0]}, D={emb.shape[1]})")

    # ====== Decode ======
    decoded_logits = fb.decode(embeddings)
    if isinstance(decoded_logits, np.ndarray):
        decoded_logits = [decoded_logits]  # ensure list consistency

    print(f"[INFO] Decoded {len(decoded_logits)} sequences")

    # ====== Inspect outputs ======
    for i, logits in enumerate(decoded_logits):
        top_idx = np.argmax(logits, axis=-1)
        toks = fb.tokenizer.convert_ids_to_tokens(top_idx.tolist())
        decoded_seq = ''.join([t if len(t) == 1 else '' for t in toks])

        print(f"\n[SEQ {i}] Original: {test_seqs[i]}")
        print(f"[SEQ {i}] Decoded : {decoded_seq}")