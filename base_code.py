import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModel

def set_seed(seed=42):
    np.random.seed(seed)  # 이 부분이 pandas의 sample 함수에도 영향을 줍니다.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

train_df = pd.read_csv('./datasets/train.csv')
test_df = pd.read_csv('./datasets/test.csv')

train_df.head()

# 현재 GPU가 사용 가능한지 확인합니다.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
cuda:0

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
model = AutoModel.from_pretrained('skt/kogpt2-base-v2')
model.to(device)

model.eval()

preds_first = []
preds_second = []
score_all_list = []
with torch.no_grad():
    # 각 테스트 케이스에 대해
    for idx in tqdm(range(len(test_df))):
        row = test_df.iloc[idx]
        best_score = float('-inf')
        best_label = 0
        
        # train 데이터에서 랜덤하게 문장을 가져옵니다.
        random_row = train_df.sample(1).iloc[0]
        random_answer = random_row['label']
        random_labels = {}
        for i in range(1, 5):
            random_labels[f'sentence{i}'] = 'O' if i == random_answer else 'X'
        
        # GPT-2에게 제공할 prompt를 작성합니다.
        example_sentence = f"""
        주어진 문장이 사람이 작성한 것이 맞으면 O, 아니면 X를 반환하세요. \

        # 예시

        문장1 : {random_row['sentence1']} -> {random_labels['sentence1']} \
        문장2 : {random_row['sentence2']} -> {random_labels['sentence2']} \
        문장3 : {random_row['sentence3']} -> {random_labels['sentence3']} \
        문장4 : {random_row['sentence4']} -> {random_labels['sentence4']} \

        # 문제
        문장 :
        """        

        score_list = []
        # 각 문장에 대한 확률값을 구하고, 가장 높은 확률값을 가진 문장을 선택합니다.
        for i in range(1, 5):
            prompt = example_sentence + " " + row[f"sentence{i}"]
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                score = outputs[0][:, -1, :].max().item()
            
            score_list.append((score,i))

        score_list.sort(key=lambda x:-x[0])
        score_all_list.append(score_list)
        preds_first.append(score_list[0][1])
        preds_second.append(score_list[1][1])       


print(score_all_list)
preds = [str(pred1) + str(pred2) for pred1,pred2 in zip(preds_first,preds_second)]
preds[:5]

submit = pd.read_csv('./datasets/sample_submission.csv')
submit['label'] = preds
submit.head()

submit.to_csv('./result/baseline_submit.csv', index=False)