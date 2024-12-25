# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:52
# @author  : Mo
# @function: test units


import traceback
import os
os.environ["FLAG_MLM_ANTONYM"] = "1"

from near_synonym import mlm_synonyms, mlm_antonyms



word = "喜欢"
word_antonyms = mlm_antonyms(word, topk=5)
word_synonyms = mlm_synonyms(word, topk=5)
print("反义词:")
print(word_antonyms)
print("近义词:")
print(word_synonyms)

word_antonym = [("前", "后"),
                ("冷", "热"),
                ("高", "矮"),
                ("进", "退"),
                ("死", "活"),
                ("快", "慢"),
                ("轻", "重"),
                ("缓", "急"),
                ("宽", "窄"),
                ("强", "弱"),
                ("宽阔", "狭窄"),
                ("平静", "动荡"),
                ("加重", "减轻"),
                ("缓慢", "快速"),
                ("节省", "浪费"),
                ("分散", "聚拢"),
                ("茂盛", "枯萎"),
                ("美丽", "丑陋"),
                ("静寂", "热闹"),
                ("清楚", "模糊"),
                ("恍恍惚惚", "清清楚楚"),
                ("一模一样", "截然不同"),
                ("柳暗花明", "山穷水尽"),
                ("风平浪静", "风号浪啸"),
                ("人声鼎沸", "鸦雀无声"),
                ("勤勤恳恳", "懒懒散散"),
                ("一丝不苟", "敷衍了事"),
                ("隐隐约约", "清清楚楚"),
                ("享誉世界", "默默无闻"),
                ("相背而行", "相向而行"),
                ]
for w in word_antonym:
    w_ant = mlm_antonyms(w[0])
    print(w[0], w[1], w_ant[0][0], w_ant)




while True:
    try:
        print("请输入word: ")
        word = input()
        if word.strip():
            word_antonyms = mlm_antonyms(word)
            word_synonyms = mlm_synonyms(word)
            print("反义词:")
            print(word_antonyms)
            print("近义词:")
            print(word_synonyms)
    except Exception as e:
        print(traceback.print_exc())

