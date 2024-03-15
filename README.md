# near-synonym
>>> near-synonym, 中文反义词/近义词(antonym/synonym)工具包.

# 一、安装
## 1.1 注意事项
   默认不指定numpy版本(标准版numpy==1.20.4)
   标准版本的依赖包详见 requirements-all.txt
   
## 1.2 通过PyPI安装
   pip install near-synonym
   使用镜像源, 如：
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple near-synonym
   不带依赖安装, 之后缺什么包再补充什么
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple near-synonym --no-dependencies
   
## 1.3 模型文件
 - github项目源码自带模型文件只有1w+词向量, 完整模型文件在near_synonym/near_synonym_model, 
 - pip下载的软件包里边只有5w+词向量, 放在data目录下;
 - 完整的词向量详见[huggingface](https://huggingface.co/)网站的[Macropodus/near_synonym_model](https://huggingface.co/Macropodus/near_synonym_model), 
 - 或完整的词向量详见百度网盘分享链接[https://pan.baidu.com/s/1lDSCtpr0r2hKrGrK8ZLlFQ](https://pan.baidu.com/s/1lDSCtpr0r2hKrGrK8ZLlFQ), 密码: ff0y;



# 二、使用方式

## 2.1 快速使用, 反义词, 近义词
```python3
import near_synonym

word = "喜欢"
word_antonyms = near_synonym.antonyms(word)
word_synonyms = near_synonym.synonyms(word)
print("反义词:")
print(word_antonyms)
print("近义词:")
print(word_synonyms)
"""
反义词:
[('讨厌', 0.6954), ('不爱', 0.6714), ('偏爱', 0.6676), ('太爱', 0.6472), ('花心', 0.6421), ('在乎', 0.6395), ('好感', 0.6378), ('酷爱', 0.634)]
近义词:
[('最爱', 0.84), ('爱好', 0.8274), ('超爱', 0.8213), ('爱上', 0.8107), ('爱玩', 0.8039), ('狂爱', 0.798), ('大胆', 0.7852), ('喜欢上', 0.7826)]
请输入word:
"""
```


## 2.2 详细使用
```python3
import near_synonym

word = "喜欢"
word_antonyms = near_synonym.antonyms(word, topk=8, annk=256, annk_cpu=128, batch_size=32,
                     rate_ann=0.4, rate_sim=0.4, rate_len=0.2, rounded=4, is_debug=False)
print("反义词:")
print(word_antonyms)
# 当前版本速度很慢, 召回数量annk_cpu/annk可以调小
```


# 三、技术原理
## 3.1 技术详情
```
near-synonym, 中文反义词/近义词工具包.
流程: Word2vec -> ANN -> NLI -> Length

# Word2vec, 词向量, 使用skip-ngram的词向量;
# ANN, 近邻搜索, 使用annoy检索召回;
# NLI, 自然语言推断, 使用Roformer-sim的v2版本, 区分反义词/近义词;
# Length, 惩罚项, 词语的文本长度惩罚;
```

## 3.2 TODO
```
1. 推理加速, 训练小的NLI模型, 替换掉笨重且不太合适的roformer-sim-ft;
2. 使用大模型构建更多的NLI语料;
```

# 四、对比
## 4.1 相似度比较
| 词语           | 2016词林改进版  | 知网hownet      | Synonyms        | near-synonym   | 
|--------------|-----------------|---------------|-----------------| ----------------- |
| "轿车","汽车"    | 0.82 | 1.0 | 0.73 | 0.86 | 
| "宝石","宝物"    | 0.83 | 0.17 | 0.71 | 0.81 |
| "旅游","游历"    | 1.0 | 1.0 | 0.59 | 0.72 | 
| "男孩子","小伙子"  | 0.81 | 1.0 | 0.88 | 0.83 |
| "海岸","海滨"    | 0.94 | 1.0 | 0.68 | 0.9 | 
| "庇护所","精神病院" | 0.96 | 0.58 | 0.64 | 0.62 |
| "魔术师","巫师"   | 0.85 | 0.58 | 0.66 | 0.78 |
| "火炉","炉灶"    | 1.0 | 1.0 | 0.81 | 0.83 | 
| "中午","正午"    | 0.98 | 0.58 | 0.85 | 0.88 |
| "食物","水果"    | 0.35 | 0.14 | 0.74 | 0.82 |
| "鸟","公鸡"     | 0.64 | 1.0 | 0.67 | 0.72 | 
| "鸟","鹤"     | 0.1 | 1.0 | 0.64 | 0.81 | 
| "工具","器械"    | 0.53 | 1.0 | 0.62 | 0.75 |
| "兄弟","和尚"    | 0.37 | 0.80 | 0.59 | 0.7 |
| "起重机","器械"   | 0.53 | 0.35 | 0.61 | 0.65 |
注：2016词林改进版/知网hownet/Synonyms数据、分数来源于[chatopera/Synonyms](https://github.com/chatopera/Synonyms)。同义词林及知网数据、分数的次级来源为[liuhuanyong/SentenceSimilarity](https://github.com/liuhuanyong/SentenceSimilarity)。


# 五、参考
 - [https://ai.tencent.com/ailab/nlp/en/index.html](https://ai.tencent.com/ailab/nlp/en/index.html)
 - [https://github.com/ZhuiyiTechnology/roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim)
 - [https://github.com/liuhuanyong/SentenceSimilarity](https://github.com/liuhuanyong/SentenceSimilarity)
 - [https://github.com/yongzhuo/Macropodus](https://github.com/yongzhuo/Macropodus)
 - [https://github.com/chatopera/Synonyms](https://github.com/chatopera/Synonyms)

# Reference
For citing this work, you can refer to the present GitHub project. For example, with BibTeX:
```
@misc{Macropodus,
    howpublished = {https://github.com/yongzhuo/near-synonym},
    title = {near-synonym},
    author = {Yongzhuo Mo},
    publisher = {GitHub},
    year = {2024}
}
```

