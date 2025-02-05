# -*- coding: utf-8 -*-
"""KJNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RydVLlbx6DDLo-6NfAfdYeDaqOlhqzlK
"""

## Harsh Bari
## bari.harsh2001@gmail.com
## Gujarat, India

"""### Header Files"""

import torch
import torch.nn as nn
import torch.optim as optim
import gensim
from gensim.models import KeyedVectors
import math
from torch.utils.data import DataLoader, TensorDataset
import string
from nltk.tokenize import word_tokenize
from gensim.downloader import load
import pandas as pd
import spacy
import zipfile
from google.colab import files
import gc
#from gensim.downloader import api

torch.manual_seed(42)

"""### Enable GPU"""

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("System Running on " + str(dev))

"""### Load Word Embeddings"""

try:
  with open('wiki_model.bin', 'r') as f:
    w_model = gensim.models.KeyedVectors.load('wiki_model.bin')
except FileNotFoundError:
  wiki_model = gensim.downloader.load('glove-wiki-gigaword-100')
  #wiki_model = api.load('glove-wiki-gigaword-100')
  wiki_model.save('wiki_model.bin')
  w_model = gensim.models.KeyedVectors.load('wiki_model.bin')

def get_word_embeddings(word):
  word = word.lower()
  if word in w_model.key_to_index:
    return torch.tensor(w_model[word], device = dev)
  return torch.tensor(w_model["default"], device = dev)

"""### Process Sentence"""

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

"""### Get Sentence Subject"""

nlp = spacy.load("en_core_web_sm")

def get_sentence_subject(sentence):

  # lowering cases
  sentence = sentence.lower()

  # remove punctuations
  sentence = remove_punctuation(sentence)

  # apply pos tagging
  doc = nlp(sentence)

  # test
  # for token in doc:
  #   print(f"Token: {token.text}, POS: {token.pos_}")

  for token in doc:
    if str(token.pos_)[-1] == "N":
      return token.text

  return doc[0].text

def create_chunks(vectors):
  return len(vectors)

"""### Positional Encodding"""

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

"""### Combine Claim and Justification"""

def combine(C, J):

  n = C.shape[0]

  C_flat = C.view(-1)
  J_flat = J.view(-1)

  new_vector = torch.empty(2*n, dtype = C.dtype)
  new_vector[0::2] = C_flat
  new_vector[1::2] = J_flat

  return new_vector.view(-1, 1)

"""### Add And Normalize"""

def add_and_normalize(sequence):
  sum_vec = sum(sequence)

  norm = torch.norm(sum_vec)

  if norm == 0:
    norm = 1

  return sum_vec / norm

"""### Train QKV Matrices"""

def chunked_avgerage(chunks, inputs):
  assert sum(chunks) == inputs.size(0), "The sum of chunks must equal the first dimension of inputs."

  output = []
  start_idx = 0

  for chunk_size in chunks:
    end_idx = start_idx + chunk_size
    chunk_avg = inputs[start_idx:end_idx].mean(dim=0)
    output.append(chunk_avg)
    start_idx = end_idx

  return torch.tensor(torch.stack(output))

def train_QKV(inputs, targets, chunks, query, key, value, num_epochs, learning_rate):

  # Adam Optimizer
  optimizer = optim.Adam([query, key, value], lr=learning_rate)

  # Mean Squared Error
  loss_fn = nn.MSELoss()
  final_loss = 0

  for epoch in range(num_epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()

    # Compute Q, K, V
    Q = inputs @ query
    K = inputs @ key
    V = inputs @ value

    # Attention scores (Q.K^T) and softmax normalization
    attention_scores = Q @ K.T
    attention_weights = torch.softmax(attention_scores, dim=1)

    # Weighted sum of values
    output2 = attention_weights @ V
    #output = chunked_avgerage(chunks, output2)

    output = []
    start = 0
    for c in chunks:
        output.append(output2[start:start + c].mean(dim=0))
        start += c


    output = torch.stack(output)

    # Compute loss
    loss = loss_fn(output, targets)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    final_loss += epoch_loss

    #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
  print("Total Loss: " + str(final_loss/num_epochs))
  return query, key, value

"""### Get Self Attention Embeddings from Query, Key and Value"""

def get_self_attention(query, key, value, vectors):

  new_vectors = []

  for vec in vectors:
    q = vec @ query
    k = vec @ key
    v = vec @ value

    output = (q @ k.T) @ V
    new_vectors.append(output)

  return torch.stack(new_vectors)

"""### Seralize Input"""

def seralize_input(input_vectors):
  return torch.cat(input_vectors, dim=0)

"""### Training Claim QKV Matrices"""

# Load Claims

dataset = pd.read_json('/content/politifact_factcheck_data.json', lines = True)
dataset['statement'].head()

gc.collect()

c_query = torch.randn(100, 100, device=dev, requires_grad=True)
c_key = torch.randn(100, 100, device=dev, requires_grad=True)
c_value = torch.randn(100, 100, device=dev, requires_grad=True)

c_learning_rate = 0.001
c_epoches = 20

chunks = 1000
total_claims = 21146

start = 0
while start < total_claims:
  print("Processing " + str(round(start*100/total_claims, 2)) + "% of the chunk")
  if start == 21000:
    end = total_claims
  else:
    end = start + chunks

  c_inputs = list(map(process_sentence, dataset.iloc[start:end]['statement']))
  c_targets = list(map(get_word_embeddings, list(map(get_sentence_subject, dataset.iloc[start:end]['statement']))))
  c_chunks = torch.tensor(list(map(create_chunks, c_inputs)))

  c_query, c_key, c_value = train_QKV(seralize_input(c_inputs), torch.stack(c_targets), c_chunks, c_query, c_key, c_value, c_epoches, c_learning_rate)

  start += chunks
  print("")

# turn off gradients
c_query.requires_grad_(False)
c_key.requires_grad_(False)
c_value.requires_grad_(False)

# save matrices
torch.save(c_query, "c_query.pt")
torch.save(c_key, "c_key.pt")
torch.save(c_value, "c_value.pt")

# to load them
# c_query = torch.load("c_query.pt")

# download
files.download("c_query.pt")
files.download("c_key.pt")
files.download("c_value.pt")

"""### Training Justification QKV Matrices"""

# Load Justifications

with zipfile.ZipFile("/content/top_justifications.zip", 'r') as zip_ref:
    zip_ref.extractall("/content")

# Read Files

def get_justifications(start, end):
  sentences = []

  for i in range(start, end):
    file_path = f"/content/top_justifications/top_justification_{i}.txt"


    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
      lines = file.readlines()

    sentences.extend(lines[1:])

  df = pd.DataFrame(sentences, columns=["sentence"])

  # Clean Whitespace and Newlines
  df["sentence"] = df["sentence"].str.strip()

  return df

gc.collect()

j_query = torch.randn(100, 100, device=dev, requires_grad=True)
j_key = torch.randn(100, 100, device=dev, requires_grad=True)
j_value = torch.randn(100, 100, device=dev, requires_grad=True)

j_learning_rate = 0.001
j_epoches = 15

chunks = 200
total_claims = 21146

start = 0
while start < total_claims:
  print("Processing " + str(round(start*100/total_claims, 2)) + "% of input")
  if start == 21000:
    end = total_claims
  else:
    end = start + chunks

  justifications = get_justifications(start, end)
  j_inputs = list(map(process_sentence, justifications['sentence']))
  j_targets = list(map(get_word_embeddings, list(map(get_sentence_subject, justifications['sentence']))))
  j_chunks = torch.tensor(list(map(create_chunks, j_inputs)))

  j_query, j_key, j_value = train_QKV(seralize_input(j_inputs), torch.stack(j_targets), j_chunks, j_query, j_key, j_value, j_epoches, j_learning_rate)

  start += chunks
  print("")

# turn off gradients
j_query.requires_grad_(False)
j_key.requires_grad_(False)
j_value.requires_grad_(False)

# save matrices
torch.save(j_query, "j_query.pt")
torch.save(j_key, "j_key.pt")
torch.save(j_value, "j_value.pt")

# to load them
# j_query = torch.load("j_query.pt")

# download
files.download("j_query.pt")
files.download("j_key.pt")
files.download("j_value.pt")

"""### Train CJ Matrix"""

def train_CJ(input_data, output, CJ_Matrix, epoches = 20, learning_rate = 0.01, batch_size = 97):

  assert input_data.size(1) == 200, "Input data must have 200 features."
  assert output.dim() == 1, "Target data must be a 1D tensor."


  class CJ_ANN(nn.Module):
    def __init__(self):
      super(CJ_ANN, self).__init__()
      self.layers = nn.Sequential(
          nn.Linear(100, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 8),
          nn.ReLU(),
          nn.Linear(8, 6)
      )

    def forward(self, x):
      return self.layers(x)

  model = CJ_ANN().to(device = dev)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam([{'params': [CJ_Matrix]}, {'params': model.parameters()}], lr=learning_rate)

  # Dataset and DataLoader
  dataset = TensorDataset(input_data, output)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  # Training loop
  for epoch in range(epoches):
    for inputs, targets in dataloader:
      transformed_inputs = torch.matmul(CJ_Matrix, inputs.t()).t()

      # Forward pass
      outputs = model(transformed_inputs)
      loss = criterion(outputs, targets)

      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Print epoch info
    if (epoch + 1) % 10 == 0:
      print(f"Epoch [{epoch + 1}/{epoches}], Loss: {loss.item():.4f}")

  #return CJ_Matrix.detach(), model
  return CJ_Matrix.detach()

"""### Training CJ Matrix"""

# Map Verdict Output

map_verdict = {"true": 0, "mostly-true": 1, "half-true": 2, "mostly-false": 3, "false": 4, "pants-fire": 5}

mapped_verdict = dataset["verdict"].map(map_verdict)

output_1 = torch.tensor(mapped_verdict.tolist(), device = dev, dtype=torch.long)
output = output_1.repeat(5)

def get_all_sentences(start, end):
  C = []
  J1 = []
  J2 = []
  J3 = []
  J4 = []
  J5 = []

  for i in range(start, end):
    file_path = f"/content/top_justifications/top_justification_{i}.txt"


    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
      lines = file.readlines()

    if len(lines) < 6:
      sen_line = len(lines)

      for k in range(6-sen_line):
        lines.append("Deafult")

    C.append(lines[0].strip())
    J1.append(lines[1].strip())
    J2.append(lines[2].strip())
    J3.append(lines[3].strip())
    J4.append(lines[4].strip())
    J5.append(lines[5].strip())

  return C, J1, J2, J3, J4, J5

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

# Create Input Data To Train CJ Matrix

# Use All Trained Matrices
c_query = torch.load("c_query.pt")
c_key = torch.load("c_key.pt")
c_value = torch.load("c_value.pt")
j_query = torch.load("j_query.pt")
j_key = torch.load("j_key.pt")
j_value = torch.load("j_value.pt")

# get all claims and justifications
C, J1, J2, J3, J4, J5 = get_all_sentences(0, 21146)

C = torch.stack([sentence_to_pre_CJ_matrix(sentence, c_query, c_key, c_value) for sentence in C])
J1 = torch.stack([sentence_to_pre_CJ_matrix(sentence, j_query, j_key, j_value) for sentence in J1])
J2 = torch.stack([sentence_to_pre_CJ_matrix(sentence, j_query, j_key, j_value) for sentence in J2])
J3 = torch.stack([sentence_to_pre_CJ_matrix(sentence, j_query, j_key, j_value) for sentence in J3])
J4 = torch.stack([sentence_to_pre_CJ_matrix(sentence, j_query, j_key, j_value) for sentence in J4])
J5 = torch.stack([sentence_to_pre_CJ_matrix(sentence, j_query, j_key, j_value) for sentence in J5])

CJ1 = torch.zeros((21146, 200), device = dev)
CJ2 = torch.zeros((21146, 200), device = dev)
CJ3 = torch.zeros((21146, 200), device = dev)
CJ4 = torch.zeros((21146, 200), device = dev)
CJ5 = torch.zeros((21146, 200), device = dev)


for i in range(21146):
  CJ1[i] = combine(C[i], J1[i]).view(-1)

for i in range(21146):
  CJ2[i] = combine(C[i], J2[i]).view(-1)

for i in range(21146):
  CJ3[i] = combine(C[i], J3[i]).view(-1)

for i in range(21146):
  CJ4[i] = combine(C[i], J4[i]).view(-1)

for i in range(21146):
  CJ5[i] = combine(C[i], J5[i]).view(-1)



# Append Vectors
input_data = torch.cat((CJ1, CJ2, CJ3, CJ4, CJ5), device = dev, dim=0)

CJ_Matrix = torch.randn(100, 200, device=dev, requires_grad=True)

CJ_Matrix = train_CJ(input_data, output, CJ_Matrix, 30, 0.0005, 97)

# turn off gradients
CJ_Matrix.requires_grad_(False)

# save matrices
torch.save(CJ_Matrix, "CJ_Matrix.pt")

# to load them
# c_query = torch.load("CJ_Matrix.pt")

# to download
files.download("CJ_Matrix.pt")

"""### Get Embeddings"""

def get_CJ_vector(CJ, CJ_Matrix):
  return CJ_Matrix @ CJ

# Load All Trained Matrices
if torch.cuda.is_available():
  c_query = torch.load("gpu_c_query.pt")
  c_key = torch.load("gpu_c_key.pt")
  c_value = torch.load("gpu_c_value.pt")
  j_query = torch.load("gpu_j_query.pt")
  j_key = torch.load("gpu_j_key.pt")
  j_value = torch.load("gpu_j_value.pt")
  CJ_Matrix = torch.load("gpu_CJ_Matrix.pt")
else:
  c_query = torch.load("cpu_c_query.pt")
  c_key = torch.load("cpu_c_key.pt")
  c_value = torch.load("cpu_c_value.pt")
  j_query = torch.load("cpu_j_query.pt")
  j_key = torch.load("cpu_j_key.pt")
  j_value = torch.load("cpu_j_value.pt")
  CJ_Matrix = torch.load("cpu_CJ_Matrix.pt")


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



C = "John McCain opposed bankruptcy protections for families 'who were only in bankruptcy because of medical expenses they couldn't pay.'"
J1 = "specifically he noted mccain opposition to an effort to exempt from the law individuals whose medical expenses pushed them into bankruptcy"
J2 = "john mccain support for a law that made it more difficult for personal bankruptcy filers to escape debts that they could repay"
J3 = "and when i fought against the credit card industry bankruptcy bill that made it harder for working families to climb out of debt he supported it and he even opposed exempting families who were only in bankruptcy because of medical expenses they could pay"
J4 = "when he had the chance to help families avoid falling into debt john mccain sided with the credit card companies"
J5 = "because obama correctly cites mccain vote on an effort to narrow the bankruptcy law reach we judge his statement true"

vector = KJNN_predict(C, J1, J2, J3, J4, J5)

print(vector)

print(type(vector))

print(vector.shape)