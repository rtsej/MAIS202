import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn import metrics 
import matplotlib.pyplot as plt 

# GPU
device = torch.device("cpu")

df = pd.read_csv("Processed_Political_Bias_with_BERT.csv")

# Split the dataset
train_text, temp_text, train_labels, temp_labels = train_test_split(df['Text'], df['Bias'], random_state=2018, test_size=0.3, stratify=df['Bias'])
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)

# Load BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences in the training, validation, and test sets
tokens_train = tokenizer.batch_encode_plus(train_text.tolist(), max_length=128, padding=True, truncation=True, return_tensors='pt')
tokens_val = tokenizer.batch_encode_plus(val_text.tolist(), max_length=128, padding=True, truncation=True, return_tensors='pt')
tokens_test = tokenizer.batch_encode_plus(test_text.tolist(), max_length=128, padding=True, truncation=True, return_tensors='pt')

# Apply LabelEncoder to convert string labels to numeric labels
label_encoder = LabelEncoder()

# Fit the encoder on the 'Bias' column (the column that contains the categorical labels)
train_labels = label_encoder.fit_transform(train_labels)  # Only fit and transform on the train labels
val_labels = label_encoder.transform(val_labels)  # Apply the same transformation to validation labels
test_labels = label_encoder.transform(test_labels)  # Apply the same transformation to test labels

# Now, convert the labels to tensors (ensure they match the number of samples in the train/val/test sets)
train_y = torch.tensor(train_labels, dtype=torch.long)
val_y = torch.tensor(val_labels, dtype=torch.long)
test_y = torch.tensor(test_labels, dtype=torch.long)

# Prepare the input tensors for train, validation, and test sets
train_seq = tokens_train['input_ids']
train_mask = tokens_train['attention_mask']

val_seq = tokens_val['input_ids']
val_mask = tokens_val['attention_mask']

test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']



# Now that the shapes match, we can create the TensorDataset
train_data = TensorDataset(train_seq, train_mask, train_y)
val_data = TensorDataset(val_seq, val_mask, val_y)
test_data = TensorDataset(test_seq, test_mask, test_y)

# Define batch size
batch_size = 32

# Create samplers and dataloaders for training and validation
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

for param in bert.parameters():
    param.requires_grad= False
import torch
import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 5)


        # Keep LogSoftmax if using NLLLoss
        self.softmax = nn.LogSoftmax(dim=1)
      


    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply LogSoftmax before returning
        x = self.softmax(x)
        
        return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
#print(torch.cuda.is_available())
#pusht the model to GPU
device = torch.device("cpu")


#optimizer from hugging face transformers
from transformers import AdamW

#define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

num_classes = len(np.unique(train_labels))
print(f"Number of classes: {num_classes}")


from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced', 
                                    classes=np.unique(train_labels), 
                                    y=train_labels)

w1 = 0.2  # Less frequent class gets a higher weight
w2 = 0.8  # More frequent class gets a lower weight

#loss_fn = nn.NLLLoss(weight=torch.tensor([w1, w2], dtype=torch.float))


#loss_fn = nn.NLLLoss(weight=torch.tensor([w1, w2]))  # Should match 2 classes
#print("Class Weights:", class_weights)

# converting list of class weights to a tensor
weights = torch.tensor([0.2, 0.8, 0.5, 0.6, 0.9], dtype=torch.float).to(device)
cross_entropy = nn.NLLLoss(weight=weights)


# push to GPU
weights = weights.to(device)



#define the loss function
cross_entropy=nn.NLLLoss(weight=weights)

#number of training epochs
epochs = 20

def train():
    
    model.train()
    total_loss, total_accuracy = 0,0
    
    #empty list to save model predictions
    total_preds=[]
    
    #iterate over batches
    for step, batch in enumerate(train_dataloader):
        
        # progress update after after every 50 batches
        if step % 50 ==0 and not step ==0:
            print(' Batch {:>5,} of {:>5,}.'.format(step, len(train_dataloader)))

            
        #push the batch to gpu
            
        batch = [r.to(device) for r in batch]
        
        sent_id, mask, labels= batch
        
        # clear previously calculated gradients
        model.zero_grad()
        
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        
        # compute the loss between actual and predicted values
        loss=cross_entropy(preds, labels)
        
        # add on the total loss
        total_loss = total_loss+loss.item()
        
        #backward pass to calculate the gradients
        loss.backward()
        
        #clip the gradient to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        
        #update parameters
        optimizer.step()
        
        #model predicitons are stored on GPU. So, push it to cpu
        preds=preds.detach().cpu().numpy()
        
    # append the model predicitons
    print("Sample prediction logits:", preds[0])
    total_preds.append(preds)
    
    #compute the training loss of the epoch
    avg_loss=total_loss/len(train_dataloader)
    
        # predicitions are in the form of (no. of batches, size of batch, no. of classes).
    total_preds = np.concatenate(total_preds, axis=0)
    
    # returns the loss and predictions

    return avg_loss, total_preds
            

import numpy as np

from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 3, 4]
y_pred = [3, 3, 3, 3, 3, 3]

# Using zero_division=0 will treat undefined precision/recall as 0
print(classification_report(y_true, y_pred, zero_division=0))

# Using zero_division=1 will treat undefined precision/recall as 1
print(classification_report(y_true, y_pred, zero_division=1))

def evaluate():
    print("\nEvaluating...")
    
    model.eval()  # set model to evaluation mode
    total_loss = 0
    total_accuracy = 0
    total_preds = []
    y_true = []  # Initialize empty list for true labels
    y_pred = []  # Initialize empty list for predicted labels
    
    # iterate over batches
    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
        
        print(' Batch{:>5,} of {:>5,}.'.format(step, len(val_dataloader)))
        
        batch = [t.to(device) for t in batch]
        
        sent_id, mask, labels = batch
        
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()  # Convert labels to numpy array
            
            total_preds.append(preds)
            y_true.extend(labels)  # Append true labels to y_true
            y_pred.extend(np.argmax(preds, axis=1))  # Get predicted class with highest probability
        
    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    
    # Print classification report
    print(classification_report(y_true, y_pred, zero_division=1))
    
    return avg_loss, total_preds


# set initial loss to infinite
best_valid_loss=float('inf')

#defining epochs
epochs = 40

# empty lists to store training and validaiton loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):
    
    print('Epoch {:} / {:}'.format(epoch+1, epochs))
    
    #train model
    train_loss, _=train()
    
    #evaluate model
    valid_loss, _=evaluate()
    
    #save the best model
    if valid_loss<best_valid_loss:
        best_valid_loss=valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
     
     #append traininf and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
    
#load weights of best modelpath = 'saved_weights.pt'model.load_state_dict(torch.load(path))


#get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

print(preds)
# model's performance
preds = np.argmax(preds, axis=1)
print()
print(preds)
print(classification_report(test_y, preds))
from sklearn.metrics import classification_report

print(test_y)
print()


# Assuming y_true and y_pred are your true labels and predicted labels
print(classification_report(y_true, y_pred, zero_division=1))
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

MSE = np.square(np.subtract(y_true, y_pred)).mean()

#print(MSE)