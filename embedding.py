# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from konlpy.tag import Twitter, Komoran
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from collections import Counter

class Embed_in_word():

    def __init__(self):
        super(Embed_in_word, self).__init__()

        self.twi = Twitter()
        # 조사, 어미, 알파벳, 숫자, 구두점, 접사, 외국어 한자
        self.tagger_list_tw = ['Josa', 'Eomi', 'Alpha', 'Number', 'Punctuation', 'Suffix', 'Foreign']

    def morpheme(self, data: list, max_length: int):
        # input : 데이터 전체 리스트
        # return : 형태소 분석된 데이터 리스트(리스트별 한문장), 형태소 분석된 데이터 전체 리스트

        senten = []
        into_sent = []
        pos_max_len = 0
        docs = []
        docs_for_vec = []

        for da in data:  # 한문장
            poses = self.twi.pos(da, norm=True, stem=True)  # normalization & stemming
            # print(poses)

            for i in range(len(poses)):
                if not i == len(poses) - 1:
                    if not poses[i][1] in self.tagger_list_tw and len(poses[i][0]) != 1:  # 특정 형태소 제거 & 한글자 제거
                        senten.append(poses[i][0] + '/' + poses[i][1])
                else:
                    if len(senten) > pos_max_len:  # 최대길이 문장 구하기
                        pos_max_len = len(senten)
                    into_sent = list(senten)
                    senten.clear()

            docs.append(into_sent)
            docs_for_vec.extend(into_sent)
        print('POS tagging success!!')
        print('max length of sentence : ', pos_max_len)
        return docs, docs_for_vec, pos_max_len


