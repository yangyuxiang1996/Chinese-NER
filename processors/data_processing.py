#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-07-06 22:59:30
LastEditors: yangyuxiang
LastEditTime: 2021-07-07 16:33:00
FilePath: /Chinese-Product-Search/processors/data_processing.py
Description:
'''
import jieba
import jieba.posseg
import pandas as pd
import os
import random


basedir = '/Volumes/yyx/projects/Chinese-Product-Search'
print(jieba.lcut("我的cf卡呢？"))
jieba.load_userdict(os.path.join(basedir, 'datasets/product.txt'))
print(jieba.lcut("我的cf卡呢？"))


def do_segment_all(sentence, product):
    sentence = sentence.strip()
    outstr = []
    sentence_seged = jieba.posseg.cut(sentence)
    for x in sentence_seged:
        if x.word in product:
            outstr.append(x.word)
    return ",".join(outstr)


def do_segment_adj(sentence):
    sentence = sentence.strip()
    outstr = []
    sentence_seged = jieba.posseg.cut(sentence)
    for x in sentence_seged:
        if x.flag in ['a', 'ad', 'ag', 'al', 'an']:
            outstr.append(x.word)
    return ",".join(outstr)


def encode(words, fenci_product, fenci_adj, fenci_brand):
    """
    按照词性，对word转为ner编码
    """
    encodings = []
    # code映射为编码
    for word in words:
        if word is None:
            continue
        word_encoding = []
        word_label = "other"
        if word in fenci_product:
            word_label = "PRODUCT"
        elif word in fenci_brand:
            word_label = "BRAND"
        elif word in fenci_adj:
            word_label = "ADJ"

        if word_label == "other":
            word_encoding.extend([x + "\t" + "O" for x in word])
        else:
            word_encoding = [word[0] + "\t" + "B-" + word_label]
            for i in range(1, len(word)):
                word_encoding.append(word[i] + "\t" + "I-" + word_label)

        encodings.extend(word_encoding)

    return encodings

def output_bio(data, path):
    """
    将bio格式的数组写入到file中
    """
    count = 0
    with open(path, 'w', encoding='utf8') as f:
        for word in data:
            for char in word:
                f.write(char + '\n')
            f.write('\n')
            count += 1
    print('已将{}个词写入到{}'.format(count, path))


def generate_data(data):
    VALID_RATIO = 0.2
    TEST_RATIO = 0.2
    bio_result_list = []
    for _, row in data.iterrows():
        try:
            fenci_product = row["product"].split(',')
            fenci_brand = row["品牌"].split(',')
            fenci_adj = row["adj"].split(',')
            querys = jieba.lcut(row['title'].strip())
            temp_res = encode(querys, fenci_product, fenci_adj, fenci_brand)
            if temp_res == None:
                continue
            bio_result_list.append(temp_res)
        except Exception as e:
             print(row)

    random.shuffle(bio_result_list)
    valid_count = int(len(bio_result_list) * VALID_RATIO)
    test_count = int(len(bio_result_list) * TEST_RATIO)
    valid_list = bio_result_list[:valid_count]
    test_list = bio_result_list[valid_count:valid_count + test_count]
    train_list = bio_result_list[valid_count + test_count:]
    # 写入文件
    output_bio(train_list, os.path.join(basedir, 'datasets/train_NER_0707.txt'))
    output_bio(valid_list, os.path.join(basedir, 'datasets/valid_NER_0707.txt'))
    output_bio(test_list, os.path.join(basedir, 'datasets/test_NER_0707.txt'))


def main():
    product = []
    with open(os.path.join(basedir, 'datasets/product.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            product.append(line)

    data_jd = pd.read_json(os.path.join(basedir, 'datasets/kb_jd_jsonl.txt'), lines=True)
    data_sn = pd.read_json(os.path.join(basedir, 'datasets/kb_sn_jsonl.txt'), lines=True)
    data = pd.concat([data_jd[["title", "品牌"]], data_sn[["title", "品牌"]]], axis=0)
    print(data.columns)
    print(data.shape)
    print(data.head(10))
    
    data['product'] = data['title'].apply(lambda x: do_segment_all(x, product))
    data['adj'] = data['title'].apply(do_segment_adj)
    print(data[['title','品牌', 'product', 'adj']].head(10))

    generate_data(data)



if __name__ == '__main__':
    main()
    
