import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification
)
from tokenizer import KoBertTokenizer

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
}

def show_report(labels, preds):
    return classification_report(labels, preds, suffix=True)

def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)

def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }