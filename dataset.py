# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os

import numpy as np
from torch.utils.data import Dataset
from gensim.models.word2vec import Word2Vec
import torch
from embedding import Embed_in_word

model = Word2Vec.load('/home/model_files/review_model.100.10.30')
# model = Word2Vec.load('review_model.100.10.30')


class MovieReviewDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """

    def __init__(self, dataset_path: str, max_length: int):
        """
        initializer

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')


        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt', encoding='utf-8') as f:
            self.reviews = preprocess(f.readlines(), max_length)
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            self.labels = [np.float32(x) for x in f.readlines()]

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.reviews[idx], self.labels[idx]


def preprocess(data: list, max_length: int):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, …])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], …]) max_length가 4일 때
    """

    em = Embed_in_word()
    doc, doc_vec, pos_max_len = em.morpheme(data, max_length)
    vec_size = len(doc_vec)
    print('total size of vectors : ', vec_size)
    print('count vectors : ', len(set(doc_vec)))
    vc_size = len(model.wv.vocab)
    print('vocab size : ', vc_size)

    pos_max_len = pos_max_len + 1
    vectored = np.zeros((len(doc), pos_max_len, max_length), dtype=np.float32) #데이터 전체 벡터

    for idx, seq in enumerate(doc):
        vector = np.zeros((pos_max_len, max_length))
        i = 0
        for word in seq:
            if word in model.wv.vocab:
                vector[i, :] = np.array(model[word])
            else:
                vector[i, :] = np.zeros(max_length, dtype=np.float32)
        i += 1

        if idx % 50000 == 0:
            print(idx, seq)
        vectored[idx, :, :] = vector

    # print(vectorized_data)
    # print(np.array(vectorized_data).shape)

    return vectored
