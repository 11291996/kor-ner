# Kor_ner

참고 사이트:  [https://github.com/monologg/KoBERT-NER/blob/master/README.md#prediction](https://github.com/monologg/KoBERT-NER/blob/master/README.md#prediction)

## accelerate를 이용해 multi-gpu training and inference 구현

 

```bash
accelerate launch kobert_train.py
accelerate launch kobert_pred.py
```

해당 bash command를 수정하여 GPU 개수 수정 가능

해당 파일들을 수정해 배치 사이즈 및 hyperparameter 수정 가능 또한 다른 모델 구현 가능