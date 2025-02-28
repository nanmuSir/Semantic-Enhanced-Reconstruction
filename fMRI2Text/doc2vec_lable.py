import json
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re
import numpy as np

with open(' ', 'r') as file:
    data = json.load(file)


def preprocess_sentence_with_labels(semantics):
    semantics = semantics.replace("annotation:\n", "")
    categories = re.findall(r'(\w+): ([\w\s,]+)', semantics, re.IGNORECASE)
    words = []
    for category, value in categories:
        category_label = category.lower()
        value_words = value.split(', ')
        for word in value_words:
            #带标签
            #words.append(f"{category_label}_{word.strip().lower()}")
            #没标签
            words.append(f"{word.strip().lower()}")
    return words

# 将每条语句转换为TaggedDocument
tagged_data_with_labels = []
for i, item in enumerate(data):
    words = preprocess_sentence_with_labels(item["annotation"])
    print(words)
    tagged_data_with_labels.append(TaggedDocument(words=words, tags=[str(i)]))
#print(tagged_data_with_labels)

#训练Doc2Vec模型
model_with_labels = Doc2Vec(tagged_data_with_labels, vector_size=200, window=2, min_count=1, workers=4, epochs=40)

#获取向量表示
vectors_with_labels = [model_with_labels.dv[str(i)] for i in range(len(tagged_data_with_labels))]

#保存为numpy数组
import numpy as np
vectors_with_labels = np.array(vectors_with_labels)
np.save(' ', vectors_with_labels)