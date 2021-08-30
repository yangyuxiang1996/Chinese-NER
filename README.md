# Chinese-NER

支持：
运行入口 `run.sh`
* BiLSTM: `model_type=bilstm`
* BiLSTM-CRF: 
    `model_type=bilstm, use_crf`
* Bert: 
    `model_type=bert`
* Bert-CRF: 
    `model_type=bert, use_crf`
* Bert-BiLSTM-CRF: 
    `model_type=bert, use_crf, use_lstm`


## weibo数据集测试结果
数据描述：
* entities:
```
{   
    'SCENE': 491, 
    'TYPE': 8655, 
    'FUNC': 3106, 
    'OUTLOOK': 1222, 
    'PRICE': 538, 
    'HARD': 4377, 
    'BRAND': 19314, 
    'SYS': 338
}
```
* 训练集和测试集
```
    train.txt, num: 4792
    test.txt, num: 1198
```

| model          | train_time | epoch | batch_size | f1     | prediction | recall |
| ---            | ---        | ---   | ---        | ---    |   ---      | ---    |
| bilstm         | 3min        | 10   | 32        | 0.7877  |  0.7939    | 0.7817    |
| bilstm_crf     | 10min       | 10   | 32        | 0.7958  | 0.7937     | 0.7980    |
| bert           | 15min       | 10   | 32        | 0.8156  | 0.8062     | 0.8253    |
| bert_crf       | 22min       | 10   | 32        | 0.8214  | 0.8018     | 0.8420    |
| bert_bilstm_crf| 24min       | 10   | 32        | 0.8275  | 0.8163     | 0.8390    |