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

from near_synonym.tools import download_model_from_huggface, load_json
from near_synonym.nli_ernie_lm import QwenErnieSim
from near_synonym.nli_ernie_tc import NliErnieSim
from near_synonym.tools import sigmoid, softmax
from near_synonym.roformer import RoformerSim
from near_synonym.search import AnnoySearch


class NearSynonym:
    def __init__(self):
        self.path_near_synonym_model_dir = os.path.join(path_sys, "near_synonym/near_synonym_model")
        self.path_ci_atmnonym_synonym = os.path.join(self.path_near_synonym_model_dir, "ci_atmnonym_synonym.json")
        self.path_model_onnx = os.path.join(self.path_near_synonym_model_dir, "roformer_unilm_small.onnx")
        self.path_tokenizer = os.path.join(self.path_near_synonym_model_dir, "roformer_vocab.txt")
        self.path_ann = os.path.join(self.path_near_synonym_model_dir, "word2vec.ann")
        self.path_w2i = os.path.join(self.path_near_synonym_model_dir, "word2vec.w2i")
        self.use_ernie = True
        self.n_cluster = 32
        self.dim_ann = 100
        self.maxlen = 512
        self.threads = 1
        self.download_hf_model()
        self.load_deep_model()
        self.load_search()
        self.load_ci()

    def near_synonym(self, word, topk=5, annk=64, annk_cpu=16, batch_size=32, rate_ann=0.4,
                     rate_sim=0.4, rate_len=0.2, rounded=4, use_pre_dict=True, is_debug=False):
        """   获取近义词   """

        ###  0. 首先搜索预置词典
        if use_pre_dict:  # 使用预置的近义词/反义词词典
            word_syn_list = self.search_ci_support(word, kind="synonym")
        else:
            word_syn_list = []
        if word_syn_list:
            score_rofo_list = []
            score_ann_list = []
            word_list = []
            for word_syn in word_syn_list:
                word_list.append([word, word_syn])
                score_ann = self.similarity_ann(word, word_syn)
                score_ann_list.append(score_ann)
                if len(word_list) % batch_size == 0:
                    score_rofo_bz = self.deep_model.sim(word_list)
                    score_rofo_list.extend(score_rofo_bz)
                    word_list = []
            if word_list:
                score_rofo_bz = self.deep_model.sim(word_list)
                score_rofo_list.extend(score_rofo_bz)

        else:
            ###  1. 获取索引和词向量, 未登录词unk获取每个字向量的平均
            ann_idx, vector = self.search_word_vector(word)

            #### 注意, 如果是cpu跑, 最低annk为annk_cpu
            if not self.deep_model.use_gpu_flag():
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
                        score_rofo_bz = self.deep_model.sim(word_list)
                        score_rofo_list.extend(score_rofo_bz)
                        word_list = []
            if word_list:
                score_rofo_bz = self.deep_model.sim(word_list)
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

    def near_antonym(self, word, topk=5, annk=256, annk_cpu=96, batch_size=32, rate_ann=0.5,
                     rate_sim=0.4, rate_len=0.1, rounded=4, use_pre_dict=True, is_debug=False):
        """   获取反义词   """
        ###  0. 首先搜索预置词典
        if use_pre_dict:  # 使用预置的近义词/反义词词典
            word_syn_list = self.search_ci_support(word, kind="antonym")
        else:
            word_syn_list = []
        if word_syn_list:
            score_rofo_list = []
            score_ann_list = []
            word_list = []
            for word_syn in word_syn_list:
                word_list.append([word, word_syn])
                score_ann = self.similarity_ann(word, word_syn)
                score_ann_list.append(score_ann)
                if len(word_list) % batch_size == 0:
                    score_rofo_bz = self.deep_model.sim(word_list)
                    score_rofo_list.extend(score_rofo_bz)
                    word_list = []
            if word_list:
                score_rofo_bz = self.deep_model.sim(word_list)
                score_rofo_list.extend(score_rofo_bz)

        else:
            ###  1. 获取索引和词向量, 未登录词unk获取每个字向量的平均
            ann_idx, vector = self.search_word_vector(word)

            #### 注意, 如果是cpu跑, 最低annk为annk_cpu
            if not self.deep_model.use_gpu_flag():
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
                        score_rofo_bz = self.deep_model.sim(word_list)
                        score_rofo_list.extend(score_rofo_bz)
                        word_list = []
            if word_list:
                score_rofo_bz = self.deep_model.sim(word_list)
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
            # score = score_ann * rate_ann
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
        score_rofo = self.deep_model.sim([(word1, word2)])[0]
        score_rofo = float(score_rofo)
        score_ann = float(score_ann)
        score = (score_ann * rate_ann + score_rofo * rate_sim + score_len * rate_len) / (rate_ann + rate_sim + rate_len)
        score_rofo = round(score_rofo, rounded)
        score_ann = round(score_ann, rounded)
        score = round(score, rounded)
        res = (word1, word2, score_ann, score_rofo, score)
        if not is_debug:
            res = res[-1]
        return res

    def similarity_ann(self, word1, word2, rounded=4, is_debug=False):
        """   词语的相似度计算-ann   """
        ann_idx_1, vector_1 = self.search_word_vector(word1)
        ann_idx_2, vector_2 = self.search_word_vector(word2)
        # score_len = 1 - abs((len(word1) - len(word2)) / (len(word1) + len(word2)))
        score_ann = vector_1.dot(vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        return score_ann

    def search_ci_support(self, word, kind="antonym"):
        """   获取反义词/同义词   """
        if kind == "antonym":
            word_kind = self.atmnonym_dict.get(word, [])
        else:
            word_kind = self.synonym_dict.get(word, [])
        return word_kind

    def search_word_vector(self, word, kind="word"):
        """   获取词向量   """
        ann_idx = 0
        ###  1. 获取索引和词向量, 未登录词unk获取每个字向量的平均
        if word and word in self.ann_w2i:
            ann_idx = self.ann_w2i.get(word, 0)
            vector = self.ann_model.get_vector(ann_idx)
            vector = np.array(vector)
        elif not word:
            ann_idx = self.ann_w2i.get("我", 0)
            vector = self.ann_model.get_vector(ann_idx)
            vector = np.array(vector)
        else:  # unk word will average char-vector
            vector = []
            if kind == "word":
                token_list = list(self.cut_bidi(word=word))
            else:
                token_list = list(word)
            for token in token_list:
                ann_idx = self.ann_w2i.get(token, 0)
                vector_i = self.ann_model.get_vector(ann_idx)
                vector_i = np.array(vector_i) * len(token) / (len(word) + 1)
                vector.append(vector_i)
            # vector = np.array(vector).mean(axis=0)
            vector = np.array(vector).sum(axis=0)
        return ann_idx, vector

    def cut_forward(self, word, len_max=7):
        """   正向最大切词   """
        len_sen = len(word)
        i = 0
        while i < len_sen:  # while判断条件
            flag = False  # flag标志位,确定有没有在字典里边的单字词或多字词
            for j in range(min(len_sen + 1, i + len_max), -i, -1):  # 遍历从当前字到句子末尾可能成词的部分, 从最后i+len_max算起
                word_maybe = word[i:j]  # 正向可能成词的语
                if word_maybe in self.ann_w2i:  # 是否在字典里边
                    i = j  # 成词前标志i向后移动
                    flag = True  # flag标志位变化
                    yield word_maybe
                    break  # 成词则跳出循环
            if not flag:  # 未选中后单个字的情况
                yield word[i]
                i += 1

    def cut_reverse(self, word, len_max=7):
        """   反向最大切词   """
        len_sen = len(word)
        i = len_sen
        res = []
        while i > 0:  # while判断条件
            flag = False  # flag标志位,确定有没有在字典里边的单字词或多字词
            for j in range(max(0, i - len_max), i):  # 遍历从句子末尾向前可能成词的部分, 从最后i-len_max算起
                word_maybe = word[j:i]  # 正向可能成词的语
                if word_maybe in self.ann_w2i:  # 是否在字典里边
                    i = j  # 成词前标志i向后移动
                    flag = True  # flag标志位变化
                    res.append(word_maybe)
                    # yield word_maybe
                    break  # 成词则跳出循环
            if not flag:  # 未选中后单个字的情况
                i -= 1
                # yield word[i]
                res.append(word[i])
        for i in range(len(res) - 1, -1, -1):
            yield res[i]
        # return res

    def cut_bidi(self, word, len_max=7):
        """   最大双向词典切词, 即最大正向切词与最大反向切词合并, 选择词数小的那个返回   """
        res_forward = self.cut_forward(word=word, len_max=len_max)
        res_reverse = self.cut_reverse(word=word, len_max=len_max)
        res_forward_list = list(res_forward)
        res_reverse_list = list(res_reverse)
        len_res_forward = len(res_forward_list)
        len_res_reverse = len(res_reverse_list)
        if len_res_forward >= len_res_reverse:
            for rrl in res_reverse_list:
                yield rrl
        else:
            for rfl in res_forward_list:
                yield rfl

    def download_hf_model(self):
        """  从hf国内镜像加载数据   """
        if os.path.exists(self.path_near_synonym_model_dir) \
            and os.path.exists(self.path_ci_atmnonym_synonym) \
            and os.path.exists(self.path_model_onnx) \
            and os.path.exists(self.path_tokenizer) \
            and os.path.exists(self.path_ann) \
            and os.path.exists(self.path_w2i):
            pass
        else:
            # dowload model from hf
            download_model_from_huggface()

    def load_deep_model(self):
        """  下载深度学习模型   """
        if "qwen" in self.path_model_onnx:
            self.deep_model = QwenErnieSim(  # LLM-synonym-atmnonym相似度, 带一点否定判断
                path_model_onnx=self.path_model_onnx,
                path_tokenizer=self.path_tokenizer,
                threads=self.threads,
                maxlen=self.maxlen,
            )
        elif "ernie" in self.path_model_onnx:
            self.deep_model = NliErnieSim(  # NIL相似度, 带一点否定判断
                path_model_onnx=self.path_model_onnx,
                path_tokenizer=self.path_tokenizer,
                threads=self.threads,
                maxlen=self.maxlen,
            )
        else:
            self.deep_model = RoformerSim(  # Roformer相似度, 带一点否定判断
                path_model_onnx=self.path_model_onnx,
                path_tokenizer=self.path_tokenizer,
                threads=self.threads,
                maxlen=self.maxlen,
            )

    def load_search(self):
        """   下载Annoy索引模型   """
        self.ann_model = AnnoySearch(dim=self.dim_ann,
                                     n_cluster=self.n_cluster)
        self.ann_model.load(self.path_ann)
        self.ann_w2i = load_json(self.path_w2i)
        # self.ann_i2w = load_json(self.path_i2w)
        self.ann_i2w = {str(v): k for k, v in self.ann_w2i.items()}

    def load_ci(self):
        """   下载预置的同义词/反义词   """
        ci_atmnonym_synonym = load_json(self.path_ci_atmnonym_synonym)
        self.atmnonym_dict = ci_atmnonym_synonym.get("atmnonym", {})
        self.synonym_dict = ci_atmnonym_synonym.get("synonym", {})


NS = NearSynonym()


if __name__ == '__main__':
    myz = 0
    # NS = NearSynonym()
    word = "黑色"
    print("反义词-use_pre_dict")
    res = NS.near_antonym(word)
    for r in res:
        print(r)
    print("#"*128)
    print("近义词-use_pre_dict")
    res = NS.near_synonym(word)
    for r in res:
        print(r)
    print("#" * 128)
    print("反义词")
    res = NS.near_antonym(word, use_pre_dict=False)
    for r in res:
        print(r)
    print("#" * 128)
    print("近义词")
    res = NS.near_synonym(word, use_pre_dict=False)
    for r in res:
        print(r)
    while True:
        try:
            print("请输入text_1: ")
            word = input()
            word = word.strip()
            if word:
                print("反义词-use_pre_dict")
                res = NS.near_antonym(word, is_debug=False)
                for r in res:
                    print(r)
                print("#" * 128)
                print("近义词-use_pre_dict")
                res = NS.near_synonym(word, is_debug=False)
                for r in res:
                    print(r)
                print("#" * 128)
                print("反义词")
                res = NS.near_antonym(word, use_pre_dict=False)
                for r in res:
                    print(r)
                print("#" * 128)
                print("近义词")
                res = NS.near_synonym(word, use_pre_dict=False)
                for r in res:
                    print(r)
        except Exception as e:
            print(traceback.print_exc())
