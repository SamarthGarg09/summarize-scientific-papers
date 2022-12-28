import numpy as np
import pandas as pd
import nltk
import re

import torch
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')

model = SentenceTransformer('all-mpnet-base-v2')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_summary(text, num_words: int=1000):
    sentences = nltk.sent_tokenize(text)
    embeddings = model.encode(sentences, show_progress_bar=False)
    try:
        sim_matrix = cosine_similarity(embeddings)
    except Exception as e:
        print(e, type(e))
        print(embeddings.shape)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i],s, i) for i,s in enumerate(sentences)), reverse=True)
    final_sents = []
    total_length = 0
    for score, sents, i in ranked_sentences:
        total_length += len(sents.split())
        if total_length < num_words:
            final_sents.append((score, sents, i))
        else:
            break

    top_k_sents = sorted(final_sents, key=lambda x: x[2])
    sents = " ".join([s[1] for s in top_k_sents])

    return sents