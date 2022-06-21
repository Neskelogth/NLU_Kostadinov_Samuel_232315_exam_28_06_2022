import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.model_selection import train_test_split

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup


# Custom dataset extending torch dataset class to 
class CustomDataset(Dataset):
    
    
    # init method stores the encodings and the masks after tokenization in torch tensors
    def __init__(self, txt, tokenizer, max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.masks = []

        for item in txt:
          encodings_dict = tokenizer(   '<|startoftext|>'+ item + '<|endoftext|>', 
                                        padding="max_length", truncation=True, 
                                        max_length=max_length)

          self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
          self.masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    # len method needed for the training
    def __len__(self):
        return len(self.input_ids)

    # get item method needed for the training
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]




data_path = 'data/IMDB Dataset.csv'
model_id = 'gpt2'
batch_size = 2
epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8
sample_every = 100
device = 'cuda'
output_dir = 'models/'

# reading the data
data = pd.read_csv (data_path, encoding='utf-8')

#dropping unused fields and NA values
fields = ['sentiment']  # IMDB dataset
# fields = ['Release Year','Title','Origin/Ethnicity','Director','Cast','Genre','Wiki Page']  # wiki plot dataset
# fields = ['title', 'categories', 'created', 'id', 'doi']  # arxiv dataset
data.drop(fields, axis=1, inplace=True)
data.dropna(inplace=True)

# copying the values 
dataset = data.review.copy()  # IMDB
# dataset = data.Plot.copy()  # wiki plots
# dataset = data.abstract.copy()  # arxiv


# getting tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_id, bos_token='<|startoftext|>', 
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>')

# creating custom dataset
dataset = CustomDataset(dataset, tokenizer, max_length=768)

# discarding test size and taking train_size
test_size = 100
train_size = len(dataset) - test_size
train_dataset, test_dataset = train_test_split(dataset, train_size=train_size/len(dataset), shuffle=False)

# Creating datasloader, used in training
train_dataloader = DataLoader(
            train_dataset
            sampler = SequentialSampler(train_dataset),
            batch_size = batch_size
        )


# getting model
model = GPT2LMHeadModel.from_pretrained(model_id)
# resize needed since special tokens are added
model.resize_token_embeddings(len(tokenizer))

# creatin optimizer and scheduler
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)


model = model.to(device)

# training loop
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    
    # used later
    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        
        # computing outputs of the model
        batch_input_ids = batch[0].to(device)
        batch_labels = batch[0].to(device)
        batch_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  batch_input_ids,
                          labels=batch_labels, 
                          attention_mask = batch_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  
        
        batch_loss = loss.item()
        total_train_loss += batch_loss

        # output every sample_every examples
        if step % sample_every == 0 and not step == 0:

            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}'.format(step, len(train_dataloader), batch_loss))

           #backpropagation
        loss.backward()

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)       

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        

# saving model
name = 'model_IMDB.pth'
# name = 'model_ArXiv.pth'
# name = 'model_wiki_plots.pth'
torch.save_pretrained(model, output_dir + name)