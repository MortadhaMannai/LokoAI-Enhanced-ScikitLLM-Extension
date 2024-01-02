import time
from typing import List

key = "your_key"
name = "your_org_id"



from skllm.config import SKLLMConfig
SKLLMConfig.set_openai_key(key)
print("ok key")
SKLLMConfig.set_openai_org(name)
print("ok org")

from skllm import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset

# demo sentiment analysis dataset
# labels: positive, negative, neutral
X, y = get_classification_dataset()
print(X)

print("get classification df")
print(y)

clf = ZeroShotGPTClassifier(openai_model = "gpt-3.5-turbo")
print("start fit")
t = time.time()
# clf.fit(X, y)
clf.fit(None, ['positive', 'negative', 'neutral'])
f = time.time()
print(f-t)
print("start pred")
labels = clf.predict(X)
print(labels)
#
#
#
