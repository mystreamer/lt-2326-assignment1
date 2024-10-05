import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
from torcheval.metrics import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassConfusionMatrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)

# Append system path
sys.path.append('../')

from modules.ThaiOCRDataset import ThaiOCRDataset
from modules.ThaiOCRNN import ThaiOCRNN
from modules.TrainingConfig import TrainingConfig
from modules.utils import detect_platform

parser = argparse.ArgumentParser(description='Train the ThaiOCR model.')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training.')
parser.add_argument('--model_name', type=str, default='thaiocr', help='Name of the model to save.')
args = parser.parse_args()

CUDA_NUM = TrainingConfig.cuda_num
BATCH_SIZE = args.batch_size if args.batch_size else  TrainingConfig.batch_size
tf = TrainingConfig
tf.device = torch.device(detect_platform(CUDA_NUM))
DEVICE = tf.device
DTYPE = tf.dtype

MODEL_DIR_PATH = '../models/'
with open(f'{ MODEL_DIR_PATH }{ args.model_name }.ohe', "rb") as f:
    data = pickle.load(f)
    ohe = data["encoder"]
    charcode_char_map = data["charmap"]
LABEL_NUMBER = ohe.categories_[0].shape[0]

inference_df = pd.read_csv("./inference_data.csv")
inference_df = inference_df[inference_df['type'] == 'test'].rename(columns={"char_code": "label", "fpath": "image_path"})[['image_path', 'label']]
test_loader = torch.utils.data.DataLoader(ThaiOCRDataset(inference_df, ohe), batch_size=BATCH_SIZE)

num_features = BATCH_SIZE * tf.image_resize[0] * tf.image_resize[1]  # Set dimension of resized image + batch size for output layer
model = ThaiOCRNN(tf.image_resize, LABEL_NUMBER)
model.load_state_dict(torch.load(f'{ MODEL_DIR_PATH }{ args.model_name }.pth', weights_only=True))
model.to(DEVICE)
model.eval()

n_classes = len(ohe.categories_[0])

preds = []
labs = []

def test(model, test_loader):
    with torch.no_grad():
        for batch in test_loader:
            images, labels = tuple(t.to(DTYPE).to(DEVICE) for t in batch)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            _, labels_maxed = torch.max(labels.data, 1)

            preds.extend(predicted.tolist())
            labs.extend(labels_maxed.tolist())

test(model, test_loader)

report = classification_report(
    [charcode_char_map[ohe.inverse_transform(F.one_hot(torch.LongTensor([el]), n_classes))[0][0]] for el in labs], 
    [charcode_char_map[ohe.inverse_transform(F.one_hot(torch.LongTensor([el]), n_classes))[0][0]] for el in preds],
    digits=4,
    output_dict=True)
pp.pprint(report)
df = pd.DataFrame(report).transpose()
df.to_csv(f'{ MODEL_DIR_PATH }{ args.model_name }_classification_report.csv')

# matrix = confusion_matrix(
#     [charcode_char_map[ohe.inverse_transform(F.one_hot(torch.LongTensor([el]), n_classes))[0][0]] for el in labs], 
#     [charcode_char_map[ohe.inverse_transform(F.one_hot(torch.LongTensor([el]), n_classes))[0][0]] for el in preds])
# # import pdb; pdb.set_trace()
# sorted_labels = sorted(charcode_char_map.values())
# df = pd.DataFrame(matrix)
# df.columns = sorted_labels
# df.index = sorted_labels
# df.to_csv(f'{ MODEL_DIR_PATH }{ args.model_name }_confusion_matrix.csv')