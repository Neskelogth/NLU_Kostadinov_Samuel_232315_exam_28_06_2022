from pandas import read_csv
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config


def eval_model(model, encodings, window, max_length):
    loss = []
    # using tqdm to have a progress bar
    for i in tqdm(range(0, encodings.input_ids.size(1), window)):
        # start, end and length computation
        start = max(i + window - max_length, 0)
        end = min(i + window, encodings.input_ids.size(1))
        len = end - i
        # inputs for the network and targets
        input_ids = encodings.input_ids[:, start:end_loc].to(device)
        targets = input_ids.clone()
        targets[:, :-len] = -100
        
        # computing the outputs
        with torch.no_grad():
            out = model.forward(input_ids, labels=targets)
            l = out[0] * len
        # append the resulting loss to compute perplexity
        loss.append(l)

    # return the perplexity
    return torch.exp(torch.stack(loss).sum() / end)


model_id = 'gpt2'
model_path = 'models/model_IMDB.pth'
max_length = 100
window = int(max_length / 2)
data_path = 'data/IMDB Dataset.csv'
device = 'cuda'

# reading data from the csv files
data = read_csv(data_path, encoding='utf-8', header=0)
fields = ['sentiment']  # for IMDB dataset
# fields = ['title', 'categories', 'created', 'id', 'doi']  # arxiv dataset
# fields = ['Release Year','Title','Origin/Ethnicity','Director','Cast','Genre','Wiki Page']  # wiki plot dataset
# dropping fields not used and NA values
data.drop(fields, axis=1, inplace=True)
data.dropna(inplace=True)
n = len(data) - 100
# keeping only the test set
data.drop(index=data.index[:n], inplace=True, axis=0)

# getting tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained(model_id, 
                                              pad_token='<|pad|>', bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>'
                                              )

model = GPT2LMHeadModel.from_pretrained(model_id)
# model = GPT2LMHeadModel.from_pretrained(model_path)  # after fine tuning use this 
# resize is needed because special tokens are defined in the from_pretrained function
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# taking only the text from the remaining field
dataset = []
for index, row in data.iterrows():
    dataset.append(row['review'])  # IMDB
    # dataset.append(row['abstract'])  # arxiv 
    # dataset.append(row['Plot'])  # wiki plots

# encoding dataset with tokenizer and printing the evaluation
encodings = tokenizer(dataset, padding=True, truncation=True, return_tensors='pt')
print(eval_model(model, encodings, window, max_length, device))  # perplexity is the first field of the returned value