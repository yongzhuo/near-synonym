# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:02
# @author  : Mo
# @function: core of neg_antonym


import traceback
import sys
import os
path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_sys)
# print(path_sys)

# from tqdm import tqdm
import numpy as np

from near_synonym.tools import load_word2vec_from_format, load_json
from near_synonym.roformer import RoformerSim
from near_synonym.search import AnnoySearch


class NearSynonym:
    def __init__(self):
        path_near_synonym_model = os.path.join(path_sys, "near_synonym/near_synonym_model")
        path_near_synonym_data = os.path.join(path_sys, "near_synonym/data")
        if os.path.exists(path_near_synonym_model):
            path_model_dir = path_near_synonym_model
        else:  # mini-data
            path_model_dir = path_near_synonym_data
        self.path_model_onnx = os.path.join(path_model_dir, "roformer_unilm_small.onnx")
        self.path_tokenizer = os.path.join(path_model_dir, "roformer_vocab.txt")
        self.path_ann = os.path.join(path_model_dir, "word2vec.ann")
        self.path_w2i = os.path.join(path_model_dir, "word2vec.w2i")
        # self.path_i2w = os.path.join(path_sys, "near_synonym/near_synonym_model/word2vec.i2w")
        self.n_cluster = 32
        self.dim_ann = 100
        self.maxlen = 512
        self.threads = 1
        self.load_romformer()
        self.load_search()

    def load_romformer(self):
        self.roformer = RoformerSim(  # 相似度, 带一点否定判断
            path_model_onnx=self.path_model_onnx,
            path_tokenizer=self.path_tokenizer,
            threads=self.threads,
            maxlen=self.maxlen,
        )

    def load_search(self):
        self.ann_model = AnnoySearch(dim=self.dim_ann,
                                     n_cluster=self.n_cluster)
        self.ann_model.load(self.path_ann)
        self.ann_w2i = load_json(self.path_w2i)
        # self.ann_i2w = load_json(self.path_i2w)
        self.ann_i2w = {str(v): k for k, v in self.ann_w2i.items()}

    def search_word_vector(self, word):
        """   获取词向量   """
        ann_idx = 0
        ###  1. 获取索引和词向量, 未登录词unk获取每个字向量的平均
        if word in self.ann_w2i:
            ann_idx = self.ann_w2i.get(word, 0)
            vector = self.ann_model.get_vector(ann_idx)
            vector = np.array(vector)
        else:  # unk word will average char-vector
            vector = []
            for char in word:
                ann_idx = self.ann_w2i.get(char, 0)
                vector_i = self.ann_model.get_vector(ann_idx)
                vector.append(vector_i)
            vector = np.array(vector).mean(axis=0)
        return ann_idx, vector

    def near_synonym(self, word, topk=8, annk=64, annk_cpu=16, batch_size=32,
                     rate_ann=0.4, rate_sim=0.4, rate_len=0.2, rounded=4, is_debug=False):
        """   获取近义词   """

        ###  1. 获取索引和词向量, 未登录词unk获取每个字向量的平均
        ann_idx, vector = self.search_word_vector(word)

        #### 注意, 如果是cpu跑, 最低annk为annk_cpu
        if not self.roformer.use_gpu_flag():
            annk = min(annk, annk_cpu)

        ###  2. 近临搜索ANN, 使用annoy
        index_tops = self.ann_model.k_neighbors([vector], k=annk)[0]
        dist, idx = index_tops

        ###  3. 两词相似度(score_ann)/自然语言推理(score_rofo)/词长度惩罚(score_len)
        score_rofo_list = []
        score_ann_list = []
        word_syn_list = []
        word_list = []
        for dist_i, idx_i in zip(dist, idx):
            if idx_i != ann_idx:
                score_ann = float((2 - (dist_i ** 2)) / 2)
                word_syn = self.ann_i2w.get(str(idx_i))
                word_list.append([word, word_syn])
                score_ann_list.append(score_ann)
                word_syn_list.append(word_syn)
                if len(word_list) % batch_size == 0:
                    score_rofo_bz = self.roformer.sim(word_list)
                    score_rofo_list.extend(score_rofo_bz)
                    word_list = []
                if word_list:
                    score_rofo_bz = self.roformer.sim(word_list)
                    score_rofo_list.extend(score_rofo_bz)

        ###  4. 后处理与排序
        res = []
        for score_rofo, score_ann, word_syn in zip(score_rofo_list, score_ann_list, word_syn_list):
            score_len = 1 - abs((len(word_syn) - len(word)) / (len(word_syn) + len(word)))
            # score = (score_ann * rate_ann + score_rofo * rate_sim + score_len * rate_len) / (rate_ann + rate_sim + rate_len)
            score = score_ann * rate_ann + score_rofo * rate_sim + score_len * rate_len
            # score = score_ann * 0.618 + score_rofo * 0.382
            # score = score_ann * 0.382 + score_rofo * 0.618
            # score = (score_ann + score_rofo) / 2
            score_rofo = round(score_rofo, rounded)
            score_ann = round(score_ann, rounded)
            score = round(score, rounded)
            res.append((word_syn, score_ann, score_rofo, score))
        res_sort = sorted(iter(res), key=lambda x: x[-1], reverse=True)
        if not is_debug:
            res_sort = [(r[0], r[-1]) for r in res_sort[:topk]]
        return res_sort

    def near_antonym(self, word, topk=8, annk=256, annk_cpu=64, batch_size=32,
                     rate_ann=0.4, rate_sim=0.4, rate_len=0.2, rounded=4, is_debug=False):
        """   获取反义词   """

        ###  1. 获取索引和词向量, 未登录词unk获取每个字向量的平均
        ann_idx, vector = self.search_word_vector(word)

        #### 注意, 如果是cpu跑, 最低annk为annk_cpu
        if not self.roformer.use_gpu_flag():
            annk = min(annk, annk_cpu)

        ###  2. 近临搜索ANN, 使用annoy
        index_tops = self.ann_model.k_neighbors([vector], k=annk)[0]
        dist, idx = index_tops

        ###  3. 两词相似度(score_ann)/自然语言推理(score_rofo)/词长度惩罚(score_len)
        score_rofo_list = []
        score_ann_list = []
        word_syn_list = []
        word_list = []
        for dist_i, idx_i in zip(dist, idx):
            if idx_i != ann_idx:
                score_ann = float((2 - (dist_i ** 2)) / 2)
                word_syn = self.ann_i2w.get(str(idx_i))
                word_list.append([word, word_syn])
                score_ann_list.append(score_ann)
                word_syn_list.append(word_syn)
                if len(word_list) % batch_size == 0:
                    score_rofo_bz = self.roformer.sim(word_list)
                    score_rofo_list.extend(score_rofo_bz)
                    word_list = []
                if word_list:
                    score_rofo_bz = self.roformer.sim(word_list)
                    score_rofo_list.extend(score_rofo_bz)

        ###  4. 后处理与排序
        res = []
        for score_rofo, score_ann, word_syn in zip(score_rofo_list, score_ann_list, word_syn_list):
            score_len = 1 - abs((len(word_syn) - len(word)) / (len(word_syn) + len(word)))
            # score = (score_ann * rate_ann + (1-score_rofo) * rate_sim + score_len * rate_len) / (rate_ann + rate_sim + rate_len)
            score = score_ann * rate_ann + (1-score_rofo) * rate_sim + score_len * rate_len
            # score = score_ann * 0.618 + score_rofo * 0.382
            # score = score_ann * 0.382 + score_rofo * 0.618
            # score = (score_ann + score_rofo) / 2
            score_rofo = round(score_rofo, rounded)
            score_ann = round(score_ann, rounded)
            score = round(score, rounded)
            res.append((word_syn, score_ann, score_rofo, score))
        res_sort = sorted(iter(res), key=lambda x: x[-1], reverse=True)
        if not is_debug:
            res_sort = [(r[0], r[-1]) for r in res_sort[:topk]]
        return res_sort

    def similarity(self, word1, word2, rate_ann=4, rate_sim=4, rate_len=2, rounded=4, is_debug=False):
        """   词语的相似度计算   """
        ann_idx_1, vector_1 = self.search_word_vector(word1)
        ann_idx_2, vector_2 = self.search_word_vector(word2)
        score_len = 1 - abs((len(word1) - len(word2)) / (len(word1) + len(word2)))
        score_ann = vector_1.dot(vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        score_rofo = self.roformer.sim([(word1, word2)])[0]
        score = (score_ann * rate_ann + score_rofo * rate_sim + score_len * rate_len) / (rate_ann + rate_sim + rate_len)
        score_rofo = round(score_rofo, rounded)
        score_ann = round(score_ann, rounded)
        score = round(score, rounded)
        res = (word1, word2, score_ann, score_rofo, score)
        if not is_debug:
            res = res[-1]
        return res


NS = NearSynonym()


if __name__ == '__main__':
    myz = 0
    # NS = NearSynonym()
    res = NS.near_antonym("美女")
    for r in res:
        print(r)
    while True:
        try:
            print("请输入text_1: ")
            text_1 = input()
            if text_1.strip():
                res = NS.near_antonym(text_1.strip())
                for r in res:
                    print(r)
        except Exception as e:
            print(traceback.print_exc())

