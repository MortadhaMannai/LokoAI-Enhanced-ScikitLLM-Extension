from typing import List, Union

from loguru import logger
from skllm.models.gpt_zero_shot_clf import ZeroShotGPTClassifier, MultiLabelZeroShotGPTClassifier


def multiclass_zero_shot_pred(data: List[str], labels: List[str], model: str = "gpt-3.5-turbo"):
    logger.debug("initialize model...")
    clf = ZeroShotGPTClassifier(openai_model=model)
    clf.fit(None, labels)
    logger.debug("start prediction")
    pred = clf.predict(data)
    logger.debug("done...")
    return pred


def multilabel_zero_shot_pred(data: List[str], labels: List[str], model: str = "gpt-3.5-turbo", max_labels: int = 3):
    logger.debug("initialize model...")
    clf = MultiLabelZeroShotGPTClassifier(openai_model=model, max_labels=max_labels)
    clf.fit(None, [labels])
    logger.debug("start prediction")
    pred = clf.predict(data)
    logger.debug("done...")
    return pred


