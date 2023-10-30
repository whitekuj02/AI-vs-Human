from ignite.utils import manual_seed
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModel
from transformers import AdamW

from modules.dataloader import Datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Engine
from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch import cat

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import random_split
from ignite.metrics import Average

def set_seed(seed=42):
    np.random.seed(seed)  # 이 부분이 pandas의 sample 함수에도 영향을 줍니다.
    torch.manual_seed(seed)
    manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


train_dataset = Datasets(csv_file='./datasets/train.csv')

total_size = len(train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# random_split으로 데이터셋 분할
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])


train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
cuda:0

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model.to(device)

# # 모든 파라미터의 그래디언트 계산 비활성화
# for param in model.parameters():
#     param.requires_grad = False

# # 분류기의 파라미터만 그래디언트 계산 활성화
# for param in model.classifier.parameters():
#     param.requires_grad = True

epochs = 50
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = epochs * len(train_dataloader)

milestones_values = [
        (0, 5e-5),
        (num_training_steps, 0.0),
    ]

# PiecewiseLinear 기능을 위해 torch의 lr_scheduler 사용
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone[0] for milestone in milestones_values], gamma=0.1)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    
    for batch in train_dataloader:
        inputs = batch[0]
        labels = batch[1].to(device)

        loss_fn = CrossEntropyLoss()
        total_loss = 0

        for i in range(4):  # 4개의 문장에 대해
            input_ids = inputs[i].to(device)
            label = labels[i].long().to(device)  # 해당 문장의 레이블

            print("Input shape:", input_ids.shape) 
            outputs = model(input_ids)
            print("Output shape:", outputs.logits.shape)  # 출력 형태 확인
            print("Label shape:", label.shape) 

            loss = loss_fn(outputs.logits, label)
            total_loss += loss

        average_loss = total_loss / 4
        total_train_loss += average_loss.item()
        
        average_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    lr_scheduler.step()

    print(f"Epoch {epoch+1}/{epochs} - Training loss: {total_train_loss/len(train_dataloader)}")
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch[0]
            labels = batch[1].to(device)
        
            loss_fn = CrossEntropyLoss()
            total_loss = 0
            
            for i in range(4):  # 4개의 문장에 대해
                input_ids = inputs[i].to(device)
                label = labels[i].long().to(device)
                
                outputs = model(input_ids)
                loss = loss_fn(outputs.logits, label)
                total_loss += loss

            average_loss = total_loss / 4
            total_val_loss += average_loss.item()
            
    print(f"Epoch {epoch+1}/{epochs} - Validation loss: {total_val_loss/len(val_dataloader)}")

## test ##
model.eval()

preds = []
test_df = pd.read_csv('./datasets/test.csv')

with torch.no_grad():
    for idx in tqdm(range(len(test_df))):
        row = test_df.iloc[idx]
        logits_for_label1 = []  # 각 문장의 라벨 1에 대한 로짓값만 저장
        
        for i in range(1, 5):
            prompt = row[f"sentence{i}"]
            prompt = "[CLS] " + prompt + " [SEP]"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
            logits_for_label1.append(logits[1])  # 라벨 1에 대한 로짓값 저장

        # 확률이 가장 높은 두 개의 인덱스를 얻기 위해 np.argsort를 사용
        top2_indices = np.argsort(logits_for_label1)[-2:]
        
        # 결과는 1부터 4의 라벨을 가지므로 1을 더해줍니다.
        top2_labels = [i + 1 for i in top2_indices]
        
        # 두 라벨을 문자열로 연결
        combined_label = str(top2_labels[0]) + str(top2_labels[1])
        preds.append(combined_label)

preds = [str(pred) for pred in preds]
preds[:5]

submit = pd.read_csv('./datasets/sample_submission.csv')
submit['label'] = preds
submit.head()

submit.to_csv('./result/baseline_submit.csv', index=False)