from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pprint
import pickle
import numpy as np
import argparse

from modules.ThaiOCRDataset import ThaiOCRDataset
from modules.ThaiOCRNN import ThaiOCRNN
from modules.TrainingConfig import TrainingConfig
from modules.utils import detect_platform

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser(description='Train the ThaiOCR model.')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training.')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs for training.')
args = parser.parse_args()

# Load the CSV file via pandas
data = pd.read_csv('./train_test_split.csv')
# do reshuffling
data = data.sample(frac=1).reset_index(drop=True)
train_data = data[data['type'] == 'train'].rename(columns={"char_code": "label", "fpath": "image_path"})[['image_path', 'label']]
test_data = data[data['type'] == 'test'].rename(columns={"char_code": "label", "fpath": "image_path"})[['image_path', 'label']]
val_data = data[data['type'] == 'val'].rename(columns={"char_code": "label", "fpath": "image_path"})[['image_path', 'label']]

data
codes = data['char_code'].unique()
charcode_char_map = {k: data.loc[data['char_code'] == k, 'char'].iloc[0] for k in codes}

ohe = OneHotEncoder().fit(train_data[['label']].values)
res = ohe.transform([[77]]).toarray()
ohe.inverse_transform(res)
LABEL_NUMBER = ohe.categories_[0].shape[0]
print(LABEL_NUMBER)

tf = TrainingConfig
tf.device = torch.device(detect_platform(tf.cuda_num))
DEVICE = tf.device
DTYPE = tf.dtype
BATCH_SIZE = args.batch_size if args.batch_size else tf.batch_size
tf.epochs = args.epochs if args.epochs else tf.epochs

train_loader = torch.utils.data.DataLoader(ThaiOCRDataset(train_data, onehotencoder=ohe), batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(ThaiOCRDataset(test_data, onehotencoder=ohe), batch_size=BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(ThaiOCRDataset(val_data, onehotencoder=ohe), batch_size=BATCH_SIZE)

num_features = BATCH_SIZE * tf.image_resize[0] * tf.image_resize[1]
model = ThaiOCRNN(num_features, LABEL_NUMBER, BATCH_SIZE)
model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=tf.learning_rate, momentum=tf.momentum)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, optimizer, criterion, n_epoch=10, iter_print=100):
    prev_loss = np.inf
    for epoch in range(n_epoch):
        model.train()
        loss_r = 0.0
        count_r = 0
        validation_loss = 0.0
        for i, batch in enumerate(train_loader):
            # print(images.shape)
            # import pdb; pdb.set_trace()
            images, labels = tuple(t.to(DTYPE).to(DEVICE) for t in batch)
            optimizer.zero_grad()
            outputs = model(images)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_r += loss.item()
            count_r += 1
            if (i+1) % iter_print == 0:
                print(f"Epoch [{epoch+1}/{n_epoch}], Step [{i+1}/{len(train_loader)}], Average Loss: {loss_r/count_r:.4f}")
                loss_r = 0.0
                count_r = 0
        for i, batch in enumerate(val_loader):
            images, labels = tuple(t.to(DTYPE).to(DEVICE) for t in batch)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
        # Print loss after each epoch
        epoch_loss = loss_r / len(train_loader)
        print(f"\nEnd of Epoch {epoch+1}/{n_epoch}, Average Epoch Train Loss: {epoch_loss:.4f}")
        print(f"Average Validation Loss: {validation_loss / len(val_loader):.4f}")

        # Early stopping logic
        if prev_loss > validation_loss:
            # Save model
            torch.save(model.state_dict(), './models/thaiocr.pth')
            prev_loss = validation_loss
        elif validation_loss - prev_loss > 0.01:
            print("Early stopping!")
            break
        else:
            continue

def test(model, test_loader, criterion):
    model.eval()
    loss_r = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            # print("iteration")
            images, labels = tuple(t.to(DTYPE).to(DEVICE) for t in batch)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_r += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, labels_maxed = torch.max(labels.data, 1)
            correct += (predicted == labels_maxed).sum().item()
    avg_loss = loss_r / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

train(model, train_loader, optimizer, criterion, n_epoch=tf.epochs)
test(model, test_loader, criterion)

# Save the one-hot encoder
with open('./models/thaiocr.ohe', "wb") as f: 
    pickle.dump({ "encoder": ohe,
                  "charmap": charcode_char_map
                }, f)