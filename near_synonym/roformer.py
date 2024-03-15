# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:10
# @author  : Mo
# @function: roformer-model


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

from near_synonym.tools import sequence_padding
from near_synonym.tools import Tokenizer


class RoformerSim:
    def __init__(self, path_model_onnx="model.onnx",
                 path_tokenizer="vocab.txt",
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

    def sim(self, texts):
        """"计算text1与text2的相似度
        """
        input_ids, segment_ids = [], []
        for (t1, t2) in texts:
            for t in (t1, t2):
                x, s = self.tokenizer.encode(t, maxlen=self.maxlen)
                input_ids.append(x)
                segment_ids.append(s)
        input_ids = sequence_padding(input_ids)
        segment_ids = sequence_padding(segment_ids)
        tokens = {"Input-Segment": np.atleast_2d(segment_ids).astype(np.float32),
                  "Input-Token": np.atleast_2d(input_ids).astype(np.float32),
                  }
        # Z = self.sess.run(None, tokens)[0]
        # Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
        # return (Z[0] * Z[1]).sum()
        sim_list = self.sess.run(None, tokens)[0]
        res = []
        for idx in range(len(texts)):
            pos_1 = idx*2
            pos_2 = (idx+1)*2
            Z = sim_list[pos_1:pos_2]
            Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
            score = (Z[0] * Z[1]).sum()
            res.append(score)
        return res


if __name__ == '__main__':
    myz = 0

    # path_model_onnx = "./near_synonym_model/roformer_unilm_small.onnx"
    # path_tokenizer = "./near_synonym_model/roformer_vocab.txt"

    path_model_onnx = "./data/roformer_unilm_small.onnx"
    path_tokenizer = "./data/roformer_vocab.txt"
    threads = 1
    maxlen = 512
    model = RoformerSim(
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


