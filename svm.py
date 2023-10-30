from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load the training data
train_df = pd.read_csv("./datasets/train.csv")

train_text = []
train_labels = []
for i in range(len(train_df)):
    R = train_df.iloc[i,1:5].astype(str)
    l = train_df.iloc[i,5]
    one_hot_label = [0,0,0,0]
    one_hot_label[l-1] = 1
    for d,a in zip(R,one_hot_label):
        train_text.append(d.strip('"'))
        train_labels.append(a)


# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_text)

# Train a LinearSVM model
svm_model = LinearSVC()
svm_model.fit(train_vectors, train_labels)

# Load the test data
test_df = pd.read_csv("./datasets/test.csv")

test_text = []
for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        logits_for_label1 = []  # 각 문장의 라벨 1에 대한 로짓값만 저장
        
        tt = []
        for i in range(1, 5):
            prompt = row[f"sentence{i}"]
            tt.append(prompt)

        test_vectors = vectorizer.transform(tt)
        # Make predictions on the test data using the trained LinearSVM model
        test_predictions = svm_model.predict(test_vectors)
        
        cl = []
        for idx, pred in enumerate(test_predictions):
             if len(cl) > 2 :
                  break
             if pred == 1:
                  cl.append(idx)
        if len(cl) == 0:
            preds = "00"
        elif len(cl) == 1:
             preds = str(cl[0]) + "0"
        else:
             preds = str(cl[0]) + str(cl[1])
        test_text.append(preds)

submit = pd.read_csv('./datasets/sample_submission.csv')
submit['label'] = test_text
submit.head()

submit.to_csv('./result/svm_result.csv', index=False)