#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-30 12:35:11
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-30 13:00:20
FilePath: /Chinese-NER/datasets/webner/data.py
Description: 
'''
import os
import collections
import random

def main():
    data = []
    sentence = []
    labels = []
    entity = collections.defaultdict(int)
    with open('./finaldata.txt', 'r') as f:
        for line in f.readlines():
            if line == "\n":
                data.append([sentence, labels])
                sentence = []
                labels = []
            else:
                line = line.split("\t")
                word, label = line[0], line[1].replace('\n', '')
                if label[0] == 'B':
                    entity[label] += 1
                sentence.append(word)
                labels.append(label)
        if not sentence:
            data.append(sentence)
    print(data[0], len(data))
    print(entity)
    random.shuffle(data)

    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]

    output_file(train_data, 'train.txt')
    output_file(test_data, 'test.txt')


def output_file(data, path):
    print("writing output file to %s, num: %d" % (path, len(data)))
    with open(path, mode='w') as f:
        for line in data:
            if len(line) == 2:
                for word, label in zip(line[0], line[1]):
                    f.write('%s\t%s\n' % (word, label))
            f.write('\n')
    print("finish!")






if __name__ == '__main__':
    main()
                




