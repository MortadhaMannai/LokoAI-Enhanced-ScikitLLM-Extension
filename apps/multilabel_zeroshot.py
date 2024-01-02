import time

key = "your_key"
name = "your_org_id"



from skllm.config import SKLLMConfig
SKLLMConfig.set_openai_key(key)
print("ok key")
SKLLMConfig.set_openai_org(name)
print("ok org")
from skllm import MultiLabelZeroShotGPTClassifier
from skllm.datasets import get_multilabel_classification_dataset

X, y = get_multilabel_classification_dataset()
print(X[:2])
print("get classification df")
print(y)

# clf = MultiLabelZeroShotGPTClassifier(max_labels=3)
# clf.fit(X, y) #equivalente al modo successivo

candidate_labels = [
    "Quality",
    "Price",
    "Delivery",
    "Service",
    "Product Variety",
    "Customer Support",
    "Packaging",
    "User Experience",
    "Return Policy",
    "Product Information"
]
print(candidate_labels)
clf = MultiLabelZeroShotGPTClassifier(max_labels=3)
print("start fit")

clf.fit(None, [candidate_labels])
print("start pred")

labels = clf.predict(X)
print(labels)