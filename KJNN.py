# Name: Harsh Bari
# bari.harsh2001@gmail.com



#################################################
import torch
import gensim
from gensim.models import KeyedVectors
import string
from nltk.tokenize import word_tokenize
from gensim.downloader import load
import pandas as pd
import spacy
import math
#################################################



#################################################
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("System Running on " + str(dev))
#################################################



#################################################
try:
  with open('wiki_model.bin', 'r') as f:
    w_model = gensim.models.KeyedVectors.load('wiki_model.bin')
except FileNotFoundError:
  print("Downloading Word Embeddings")
  wiki_model = gensim.downloader.load('glove-wiki-gigaword-100')
  #wiki_model = api.load('glove-wiki-gigaword-100')
  wiki_model.save('wiki_model.bin')
  w_model = gensim.models.KeyedVectors.load('wiki_model.bin')
#################################################



#################################################
def get_word_embeddings(word):
  word = word.lower()
  if word in w_model.key_to_index:
    return torch.tensor(w_model[word], device = dev)
  return torch.tensor(w_model["default"], device = dev)
#################################################



#################################################
def get_CJ_vector(CJ, CJ_Matrix):
  return CJ_Matrix @ CJ
#################################################



#################################################
stop_words = ["a", "an", "the", "of", "in", "for", "through", "there", "be", "is", "was", "will", "and", "or", "not", "no", "on", "at", "under", "such", "that", "to", "with"]
punctuation_marks = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

def remove_punctuation(sentence):
  return ''.join(char for char in sentence if char not in punctuation_marks)


def process_sentence(sentence):

  # lowering cases
  sentence = sentence.lower()

  # remove punctuations
  sentence = remove_punctuation(sentence)

  # tokenize
  tokens = sentence.split()

  # remove stop words
  # tokens = [word for word in tokens if word not in stop_words]

  # Get embeddings for each token
  embeddings = [get_word_embeddings(token) for token in tokens]

  return torch.stack(embeddings)
#################################################



#################################################
def sentence_to_pre_CJ_matrix(sentence, query, key, value):

  # get each word embeddings
  processed_sentence = process_sentence(sentence)

  # apply positional encodding
  positional_encoding = get_positional_encodding(processed_sentence)

  # Compute Q, K, V
  Q = positional_encoding @ query
  K = positional_encoding @ key
  V = positional_encoding @ value

  # Attention scores (Q.K^T) and softmax normalization
  attention_scores = Q @ K.T
  attention_weights = torch.softmax(attention_scores, dim=1)

  # Weighted sum of values
  output = attention_weights @ V

  return add_and_normalize(output)
#################################################



#################################################
def get_positional_encodding(sequence):
  n, d_model = sequence.shape
  assert d_model == 100, "Input tensor must have 100 features (shape: [n, 100])"

  # Create the positional encoding matrix
  position = torch.arange(n, dtype=torch.float).unsqueeze(1)  # Shape: (n, 1)
  div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

  pe = torch.zeros((n, d_model), device = dev)  # Initialize positional encoding tensor
  pe[:, 0::2] = torch.sin(position * div_term)  # sin for even indices
  pe[:, 1::2] = torch.cos(position * div_term)  # cos for odd indices

  # Add positional encoding to input tensor
  output_tensor = sequence + pe
  return output_tensor
#################################################



#################################################
def combine(C, J):

  n = C.shape[0]

  C_flat = C.view(-1)
  J_flat = J.view(-1)

  new_vector = torch.empty(2*n, dtype = C.dtype)
  new_vector[0::2] = C_flat
  new_vector[1::2] = J_flat

  return new_vector.view(-1, 1)
#################################################



#################################################
def add_and_normalize(sequence):
  sum_vec = sum(sequence)

  norm = torch.norm(sum_vec)

  if norm == 0:
    norm = 1

  return sum_vec / norm
#################################################



#################################################
# Load All Trained Matrices
if torch.cuda.is_available():
  c_query = torch.load("parameters/gpu_c_query.pt")
  c_key = torch.load("parameters/gpu_c_key.pt")
  c_value = torch.load("parameters/gpu_c_value.pt")
  j_query = torch.load("parameters/gpu_j_query.pt")
  j_key = torch.load("parameters/gpu_j_key.pt")
  j_value = torch.load("parameters/gpu_j_value.pt")
  CJ_Matrix = torch.load("parameters/gpu_CJ_Matrix.pt")
else:
  c_query = torch.load("parameters/cpu_c_query.pt")
  c_key = torch.load("parameters/cpu_c_key.pt")
  c_value = torch.load("parameters/cpu_c_value.pt")
  j_query = torch.load("parameters/cpu_j_query.pt")
  j_key = torch.load("parameters/cpu_j_key.pt")
  j_value = torch.load("parameters/cpu_j_value.pt")
  CJ_Matrix = torch.load("parameters/cpu_CJ_Matrix.pt")
#################################################



#################################################
def KJNN_predict(C, J1, J2, J3, J4, J5):

  # Preprocess Sentences
  C = sentence_to_pre_CJ_matrix(C, c_query, c_key, c_value)
  J1 = sentence_to_pre_CJ_matrix(J1, j_query, j_key, j_value)
  J2 = sentence_to_pre_CJ_matrix(J2, j_query, j_key, j_value)
  J3 = sentence_to_pre_CJ_matrix(J3, j_query, j_key, j_value)
  J4 = sentence_to_pre_CJ_matrix(J4, j_query, j_key, j_value)
  J5 = sentence_to_pre_CJ_matrix(J5, j_query, j_key, j_value)


  # CJ Matrix
  CJ1 = get_CJ_vector(combine(C, J1), CJ_Matrix)
  CJ2 = get_CJ_vector(combine(C, J2), CJ_Matrix)
  CJ3 = get_CJ_vector(combine(C, J3), CJ_Matrix)
  CJ4 = get_CJ_vector(combine(C, J4), CJ_Matrix)
  CJ5 = get_CJ_vector(combine(C, J5), CJ_Matrix)

  # Final Add and Normalize
  return add_and_normalize(torch.stack([CJ1, CJ2, CJ3, CJ4, CJ5], dim = 0).squeeze(2))
#################################################