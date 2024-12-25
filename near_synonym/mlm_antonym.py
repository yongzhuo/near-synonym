# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: BERT-MLM to Antonym


from __future__ import absolute_import, division, print_function
import traceback
import time
import copy
import sys
import os
path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_sys)
print(path_sys)
from near_synonym.tools import download_model_from_huggface, load_json
from transformers import BertForMaskedLM, BertTokenizer, BertConfig
import torch


class MLM4Antonym:
    def __init__(self, path_pretrain_model_dir="",
                 path_trained_model_dir="",
                 device="cuda:0"):
        if not path_pretrain_model_dir:
            self.path_pretrain_model_dir = os.path.join(path_sys, "near_synonym/mlm_antonym_model")
            self.path_trained_model_dir = os.path.join(path_sys, "near_synonym/mlm_antonym_model")
        else:
            self.path_pretrain_model_dir = path_pretrain_model_dir
            self.path_trained_model_dir = path_trained_model_dir
        self.path_w2i = os.path.join(path_sys, "near_synonym/near_synonym_model/word2vec.w2i")
        self.flag_skip = False
        self.device = device
        self.topk_times = 5  # topk重复次数, 避免非中文的情况
        self.topk = 8  # beam-search
        self.check_or_download_hf_model()  # 检测模型目录是否存在, 不存在就下载模型
        self.load_trained_model()
        self.flag_filter_word = False  # 用于过滤词汇, [MASK]有时候可能不成词
        self.w2i = {}
        if os.path.exists(self.path_w2i):
            self.w2i = load_json(self.path_w2i)
        if not self.w2i:
            self.flag_filter_word = False

    def check_or_download_hf_model(self):
        """  从hf国内镜像加载数据   """
        if os.path.exists(self.path_pretrain_model_dir):
            pass
        else:
            # dowload model from hf
            download_model_from_huggface(repo_id="Macropodus/mlm_antonym_model")

    def load_trained_model(self):
        """   加载训练好的模型   """
        if "mlm_antonym" in self.path_trained_model_dir:
            self.tokenizer = BertTokenizer.from_pretrained(self.path_trained_model_dir)
            config = BertConfig.from_pretrained(pretrained_model_name_or_path=self.path_trained_model_dir)
            self.model = BertForMaskedLM(config)
            path_model_real = os.path.join(self.path_trained_model_dir, "pytorch_model.bin")
            model_dict_new = torch.load(path_model_real, map_location=torch.device(self.device))
            model_dict_new = {k.replace("pretrain_model.", ""): v for k, v in model_dict_new.items()}
            self.model.load_state_dict(model_dict_new, strict=False)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.path_trained_model_dir)
            self.model = BertForMaskedLM.from_pretrained(self.path_trained_model_dir)
        self.model.to(self.device)
        self.model.eval()

    def prompt(self, word, category="antonym"):
        """   组装提示词   """
        if category == "antonym":
            input_text = '"{}"的反义词是"{}"。'.format(word, len(word) * '[MASK]')
        elif category == "same":
            input_text = '"{}"的同义词是"{}"。'.format(word, len(word) * '[MASK]')
        else:
            input_text = '"{}"的近义词是"{}"。'.format(word, len(word) * '[MASK]')
            # input_text = '"{}"的相似词是"{}"。'.format(word, len(word) * '[MASK]')
            # input_text = '你觉得{}的近义词是{}。'.format(len_word * '[MASK]', word)
        return input_text

    def decode(self, input_ids):
        """   解码   """
        return self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=self.flag_skip)

    def predict(self, word, topk=8, category="antonym"):
        """   推理
        category可选: "antonym", "synonym", "same"
        """
        topk_times = self.topk_times * topk
        len_word = len(word)
        text = self.prompt(word, category=category)
        count_mask = text.count("[MASK]")
        # 对输入句子进行编码
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_ids = input_ids.to(self.device)
        input_bs = input_ids.repeat(topk, 1)
        score_bs = [0] * topk
        res = []
        # 进行预测第一个char, 保证取得topk个不一样的char, 类似beam-search
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        # 获取预测结果中的logits
        logits = outputs.logits
        # 获取[MASK]标记位置的索引（假设句子中只有一个[MASK]标记）
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        # 获取[MASK]标记位置上的预测结果
        mask_token_logits = logits[0, mask_token_index, :]
        # 获取概率最高的token及其对应的id和文本
        top5_softmax = torch.softmax(mask_token_logits, dim=-1)
        largest, indices = torch.topk(top5_softmax, topk_times, dim=1, largest=True, sorted=True)
        topk_ids = indices[0]
        topk_prob = largest[0]
        count = 0
        ### input_bs填充第一个[MASK], 获取word_list
        for idx, topk_id in enumerate(topk_ids):
            topk_id_token = self.tokenizer.decode([topk_id], skip_special_tokens=True)
            # token存在且是中文, 取topk个
            if topk_id_token.strip() and "\u4e00" <= topk_id_token <= "\u9fa5" and count <= topk-1:  # 必须是中文
                input_bs[count][mask_token_index[0]] = topk_id
                input_bs_idx = [self.tokenizer.decode([id]) for id in input_bs[count]
                                ][mask_token_index[0] + 1 - len_word: mask_token_index[0] + 1]
                score_bs[count] = torch.log(topk_prob[idx])
                score_count = torch.exp(score_bs[count])
                res.append(("".join(input_bs_idx), float(score_count.detach().cpu().numpy())))
                count += 1

        ### 进行预测第一个k(1>1)个char, 保证取得topk个不一样的char, 类似beam-search
        ### 得分取得topk个数值(torch.log), 然后重排序;
        for i in range(count_mask-1):
            # 进行预测
            with torch.no_grad():
                outputs = self.model(input_ids=input_bs)
            # 获取预测结果中的logits
            logits = outputs.logits
            # 获取[MASK]标记位置的索引（假设句子中只有一个[MASK]标记）
            mask_token_index = torch.where(input_bs == self.tokenizer.mask_token_id)[1]
            # input_bs_topk = input_bs.repeat(topk, 1)
            score_bs_topk = [0] * topk * topk
            input_bs_topk_all = []
            res = []
            for tdx in range(topk):
                # 获取[MASK]标记位置上的预测结果
                mask_token_logits = logits[tdx, mask_token_index, :]
                input_bs_tdx = copy.deepcopy(input_bs[tdx])
                input_bs_tdx_topk = input_bs_tdx.repeat(topk, 1)
                # 获取概率最高的token及其对应的id和文本
                top5_softmax = torch.softmax(mask_token_logits, dim=-1)
                largest, indices = torch.topk(top5_softmax, topk_times,
                                              dim=-1, largest=True, sorted=True)
                topk_ids = indices[0]
                topk_prob = largest[0]
                count = 0
                for jdx, topk_id in enumerate(topk_ids):
                    topk_id_token = self.tokenizer.decode([topk_id], skip_special_tokens=True)
                    if topk_id_token.strip() and "\u4e00" <= topk_id_token <= "\u9fa5" and count <= topk-1:  # 必须是中文
                        input_bs_tdx_topk[count][mask_token_index[0]] = topk_id
                        input_bs_topk_idx = [self.tokenizer.decode([id]) for id in input_bs_tdx_topk[count]
                                             ][1:mask_token_index[0] + 1]
                        score_bs_topk[tdx * topk + count] = score_bs[tdx] + torch.log(topk_prob[count])
                        score_count = torch.exp(score_bs_topk[tdx * topk + count])
                        res.append(("".join(input_bs_topk_idx[mask_token_index[0] - len_word:mask_token_index[0] + 1]),
                                          float(score_count.detach().cpu().numpy())))
                        input_bs_topk_all.append([input_bs_tdx_topk[count], score_count])
                        count += 1
            input_bs_topk_all_sort = [a[0].unsqueeze(0) for a in sorted(iter(input_bs_topk_all),
                                        key=lambda x: x[-1], reverse=True)[:topk]]
            input_bs = torch.cat(input_bs_topk_all_sort, dim=0)
        ### 词典过滤
        if self.flag_filter_word:
            res_sort = sorted(iter(res), key=lambda x: x[-1], reverse=True)
            res = [s for s in res_sort
                   if s[0] != word and s[0] in self.w2i][:topk]
            ### 如果全部不在词典就保留一个
            if not res:
                res = res_sort[:1]
        else:
            res_sort = sorted(iter(res), key=lambda x: x[-1], reverse=True)
            res = [s for s in res_sort
                   if s[0] != word][:topk]
        return res

    def near_antonym(self, word, topk=8, flag_normalize=True):
        """   获取反义词   """
        topk = topk or self.topk
        word_list = self.predict(word, topk=topk, category="antonym")
        if flag_normalize:
            score_total = sum([w[-1] for w in word_list])
            word_list = [(w[0], round(0.4 + 0.59*min(1, w[1]/score_total*2), 2)) for w in word_list]
        return word_list

    def near_synonym(self, word, topk=8, flag_normalize=True):
        """   获取近义词   """
        topk = topk or self.topk
        word_list = self.predict(word, topk=topk, category="synonym")
        if flag_normalize:
            score_total = sum([w[-1] for w in word_list])
            word_list = [(w[0], round(0.4 + 0.59*min(1, w[1]/score_total*2), 2)) for w in word_list]
        return word_list

    def near_same(self, word, topk=8, flag_normalize=True):
        """   获取同义词   """
        topk = topk or self.topk
        word_list = self.predict(word, topk=topk, category="same")
        if flag_normalize:
            score_total = sum([w[-1] for w in word_list])
            word_list = [(w[0], round(0.4 + 0.59 * min(1, w[1] / score_total * 2), 2)) for w in word_list]
        return word_list


def tet_predict():
    """   测试 predict 接口   """
    path_trained_model_dir = "./mlm_antonym_model"
    model = MLM4Antonym(path_trained_model_dir, path_trained_model_dir)
    model.flag_skip = False
    # model.topk = 16  # beam-search
    word = "喜欢"
    categorys = ["antonym", "synonym", "same"]
    for category in categorys:
        time_start = time.time()
        res = model.predict(word, category=category)
        time_end = time.time()
        print(f"{word}的{category}: ")
        for r in res:
            print(r)
        print(time_end - time_start)
    while 1:
        try:
            print("请输入：")
            word = input()
            word = word.strip()
            time_start = time.time()
            categorys = ["antonym", "synonym", "same"]
            for category in categorys:
                res = model.predict(word, category=category)
                time_end = time.time()
                print(f"{word}的{category}: ")
                for r in res:
                    print(r)
                print(time_end - time_start)
        except Exception as e:
            print(traceback.print_exc())
def tet_antonym():
    """   测试 对外函数   """
    ### 可以放训练好的模型, 也可以放开源的bert类模型(效果差一些)
    path_trained_model_dir = "./mlm_antonym_model"
    # path_trained_model_dir = "E:/DATA/bert-model/00_pytorch/MacBERT-chinese_finetuned_correction"
    # path_trained_model_dir = "E:/DATA/bert-model/00_pytorch/LLM/hfl_chinese-macbert-base"
    # path_trained_model_dir = "E:/DATA/bert-model/00_pytorch/bert-base-chinese"
    # path_trained_model_dir = "E:/DATA/bert-model/00_pytorch/chinese-roberta-wwm-ext"
    # path_trained_model_dir = "E:/DATA/bert-model/00_pytorch/pai-ckbert-base-zh"
    model = MLM4Antonym(path_trained_model_dir, path_trained_model_dir)
    model.flag_skip = False
    # model.topk = 16  # beam-search
    word = "喜欢"

    ### antonym
    time_start = time.time()
    res = model.near_antonym(word)
    time_end = time.time()
    print(f"{word}的antonym: ")
    for r in res:
        print(r)
    print(time_end - time_start)
    ### synonym
    time_start = time.time()
    res = model.near_synonym(word)
    time_end = time.time()
    print(f"{word}的synonym: ")
    for r in res:
        print(r)
    print(time_end - time_start)


    while 1:
        try:
            print("请输入：")
            word = input()
            word = word.strip()
            ### antonym
            time_start = time.time()
            res = model.near_antonym(word)
            time_end = time.time()
            print(f"{word}的antonym: ")
            for r in res:
                print(r)
            print(time_end - time_start)
            ### synonym
            time_start = time.time()
            res = model.near_synonym(word)
            time_end = time.time()
            print(f"{word}的synonym: ")
            for r in res:
                print(r)
            print(time_end - time_start)
        except Exception as e:
            print(traceback.print_exc())


### 初始化bert-mlm
MA = MLM4Antonym()


if __name__ == "__main__":
    yz = 0

    """   测试 predict 函数   """
    # tet_predict()
    """   测试 对外函数   """
    # tet_antonym()

    """   测试 初始化模型   """
    # MA.flag_filter_word = True  # 用于过滤词汇, [MASK]有时候可能不成词
    # MA.flag_skip = False  # decode的时候, 特殊字符是否跳过
    # MA.topk_times = 5  # topk重复次数, 避免非中文的情况
    # MA.topk = 8  # eg.5, 16, 32; 类似beam-search, 但是第一个char的topk必须全选

    word = "喜欢"
    ### antonym
    time_start = time.time()
    res = MA.near_antonym(word, topk=8)
    time_end = time.time()
    print(f"{word}的antonym: ")
    for r in res:
        print(r)
    print(time_end - time_start)
    ### synonym
    time_start = time.time()
    res = MA.near_synonym(word, topk=8)
    time_end = time.time()
    print(f"{word}的synonym: ")
    for r in res:
        print(r)
    print(time_end - time_start)


    while 1:
        try:
            print("请输入：")
            word = input()
            word = word.strip()
            ### antonym
            time_start = time.time()
            res = MA.near_antonym(word)
            time_end = time.time()
            print(f"{word}的antonym: ")
            for r in res:
                print(r)
            print(time_end - time_start)
            ### synonym
            time_start = time.time()
            res = MA.near_synonym(word)
            time_end = time.time()
            print(f"{word}的synonym: ")
            for r in res:
                print(r)
            print(time_end - time_start)
        except Exception as e:
            print(traceback.print_exc())


"""
喜欢的antonym: 
('厌恶', 0.77)
('讨厌', 0.72)
('憎恶', 0.56)
('反恶', 0.49)
('忌恶', 0.48)
('反厌', 0.46)
('厌烦', 0.46)
('反感', 0.45)
4.830108404159546
喜欢的synonym: 
('喜好', 0.75)
('喜爱', 0.64)
('爱好', 0.54)
('倾爱', 0.5)
('爱爱', 0.49)
('喜慕', 0.49)
('向好', 0.48)
('倾向', 0.48)
0.15957283973693848
"""


