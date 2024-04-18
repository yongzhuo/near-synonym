# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:52
# @author  : Mo
# @function: test units


import traceback
import logging

import near_synonym

w1 = "桂林"
w2 = "柳州"
score = near_synonym.sim(w1, w2)
print(w1, w2, score)

words_syn = [
    ("轿车", "汽车"),
    ("宝石", "宝物"),
    ("旅游", "游历"),
    ("男孩子", "小伙"),
    ("海岸", "海滨"),
    ("庇护所", "精神病院"),
    ("魔术师", "巫师"),
    ("中午", "正午"),
    ("火炉", "炉灶"),
    ("食物", "水果"),
    ("鸟", "公鸡"),
    ("鸟", "鹤"),
    ("工具", "器械"),
    ("兄弟", "和尚"),
    ("起重机", "器械"),
]
for w1, w2 in words_syn:
    score = near_synonym.sim(w1, w2)
    print((w1, w2, score))


while True:
    try:
        print("请输入word_1: ")
        word_1 = input()
        print("请输入word_2: ")
        word_2 = input()
        if word_1.strip() and word_2.strip():
            score = near_synonym.sim(word_1, word_2)
            print((word_1, word_2, score))
    except Exception as e:
        print(traceback.print_exc())


"""
('轿车', '汽车', 0.8623)
('宝石', '宝物', 0.8081)
('旅游', '游历', 0.7192)
('男孩子', '小伙', 0.77)
('海岸', '海滨', 0.8973)
('庇护所', '精神病院', 0.6248)
('魔术师', '巫师', 0.7787)
('中午', '正午', 0.8339)
('火炉', '炉灶', 0.8788)
('食物', '水果', 0.8205)
('鸟', '公鸡', 0.717)
('鸟', '鹤', 0.8067)
('工具', '器械', 0.7524)
('兄弟', '和尚', 0.7041)
('起重机', '器械', 0.6494)
请输入word_1: 
"""


