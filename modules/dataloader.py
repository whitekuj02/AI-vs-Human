from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

class Datasets(Dataset):
    def __init__(self, csv_file, mode="train"):
        
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 라벨은 0부터 시작하므로 1을 빼줍니다.
        label = self.data.iloc[idx, 5] - 1  # Assuming labels are 1-indexed
        labels = torch.full((4,), label, dtype=torch.long)  # [num_sentences]
        # one-hot encoding으로 라벨을 변환합니다.
        # one_hot_label = torch.zeros(4)
        # one_hot_label[label] = 1

        sentences = ["[CLS] " + self.data.iloc[idx, col] + " [SEP]" for col in range(1, 5)]
        
        # 인코딩 및 패딩
        inputs = [self.tokenizer.encode_plus(sentence, 
                                         add_special_tokens=True, 
                                         max_length=512, 
                                         padding='max_length', 
                                         truncation=True) for sentence in sentences]

        # 리스트를 텐서로 변환
        input_ids = torch.stack([torch.tensor(single_input['input_ids'][0]) for single_input in inputs])
        attention_mask = torch.stack([torch.tensor(single_input['attention_mask'][0]) for single_input in inputs])
        token_type_ids = torch.stack([torch.tensor(single_input['token_type_ids'][0]) for single_input in inputs])


        if self.mode == "train" or self.mode == "validation":
            return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}, labels
        
        elif self.mode == "test":
            return inputs
