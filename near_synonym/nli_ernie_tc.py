# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:10
# @author  : Mo
# @function: ernie-nli(zh+multi)


import traceback
import sys
import os
path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_sys)
# print(path_sys)

from onnxruntime import InferenceSession
from onnxruntime import SessionOptions
from onnxruntime import ExecutionMode
import onnxruntime
import numpy as np

from near_synonym.tools import sigmoid, softmax
from near_synonym.tools import sequence_padding
from near_synonym.tools import Tokenizer


class NliErnieSim:
    def __init__(self, path_model_onnx="ernie_nli_tc.onnx",
                 path_tokenizer="ernie_vocab.txt",
                 maxlen=512,
                 threads=1):
        self.path_model_onnx = path_model_onnx
        self.path_tokenizer = path_tokenizer
        self.threads = threads
        self.maxlen = maxlen
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self):
        """   加载分词器   """
        self.tokenizer = Tokenizer(  # 加载分词器
            token_dict=self.path_tokenizer,
            do_lower_case=True)

    def use_gpu_flag(self):
        """  搜索使用的设备, GPU or CPU   """
        provider_options_dict = self.sess.get_provider_options()
        device_name = onnxruntime.get_device()
        if "CUDAExecutionProvider" in provider_options_dict:
            flag = True
        else:
            flag = False
        return flag

    def load_model(self):
        """   加载ONNX模型   """
        options = SessionOptions()
        options.intra_op_num_threads = self.threads
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        try:
            self.sess = InferenceSession(
                providers=["CUDAExecutionProvider"],
                path_or_bytes=self.path_model_onnx,
                sess_options=options,
            )
        except Exception as e:
            print(traceback.print_exc())
            self.sess = InferenceSession(
                providers=["CPUExecutionProvider"],
                path_or_bytes=self.path_model_onnx,
                sess_options=options,
            )
            print("use CPUExecutionProvider!")

    def sim(self, texts, logits_type="SIGMOID"):
        """"计算text1与text2的相似度
        """
        input_ids, segment_ids, mask_ids = [], [], []
        for (t1, t2) in texts:
            x, s, m = self.tokenizer.encode(first_text=t1, second_text=t2, maxlen=self.maxlen, return_mask=True)
            input_ids.append(x)
            segment_ids.append(s)
            mask_ids.append(m)
        input_ids = sequence_padding(input_ids)
        segment_ids = sequence_padding(segment_ids)
        mask_ids = sequence_padding(mask_ids)
        tokens = {"token_type_ids": np.atleast_2d(segment_ids).astype(np.int64),
                  "input_ids": np.atleast_2d(input_ids).astype(np.int64),
                  "attention_mask": np.atleast_2d(mask_ids).astype(np.int64),
                  }
        res_model = self.sess.run(None, tokens)[0]

        # if logits_type.upper() == "SIGMOID":
        #     logits_numpy = sigmoid(res_model)
        # elif logits_type.upper() == "SOFTMAX":
        #     logits_numpy = softmax(res_model)
        res = [r[0] for r in res_model]
        return res


if __name__ == '__main__':
    myz = 0

    # path_model_onnx = "./near_synonym_model/roformer_unilm_small.onnx"
    # path_tokenizer = "./near_synonym_model/roformer_vocab.txt"

    path_model_onnx = "./data/ernie_nli_onnx/ernie_nli_model.onnx"
    path_tokenizer = "./data/ernie_nli_onnx/ernie_nli_vocab.txt"
    threads = 1
    maxlen = 512
    model = NliErnieSim(
        path_model_onnx=path_model_onnx,
        path_tokenizer=path_tokenizer,
        threads=threads,
        maxlen=maxlen,
    )
    print(model.sim([(u'给我推荐一款红色的车1', u'给我推荐一款黑色的车1'),
                     (u'给我推荐一款红色的车1', u'给我推荐一款黑色的车2'),
                     (u'给我推荐一款红色的车1', u'给我推荐一款黑色的车2')
                     ]))
    # print(model.sim(u'给我推荐一款红色的车', u'给我推荐一款黑色的车'))
    while True:
        try:
            print("请输入text_1: ")
            text_1 = input()
            print("请输入text_2: ")
            text_2 = input()
            if text_1.strip() and text_2.strip():
                # print(model.sim(text_1, text_2))
                print(model.sim([(text_1, text_2)]))
        except Exception as e:
            print(traceback.print_exc())


