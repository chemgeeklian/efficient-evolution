import datetime
import numpy as np
import os
import random
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
from Bio import Seq, SeqIO

np.random.seed(1)
random.seed(1)

AAs = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
]

def tprint(msg):
    print(f"{datetime.now()} | {msg}", flush=True)

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
        
def deep_mutational_scan(sequence, exclude_noop=True):
    for pos, wt in enumerate(sequence):
        for mt in AAs:
            if exclude_noop and wt == mt:
                continue
            yield (pos, wt, mt)