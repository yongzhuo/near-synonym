# [near-synonym](https://github.com/yongzhuo/near-synonym) 
[![PyPI](https://img.shields.io/pypi/v/near-synonym)](https://pypi.org/project/near-synonym/)
[![Build Status](https://travis-ci.com/yongzhuo/near-synonym.svg?branch=master)](https://travis-ci.com/yongzhuo/near-synonym)
[![PyPI_downloads](https://img.shields.io/pypi/dm/near-synonym)](https://pypi.org/project/near-synonym/)
[![Stars](https://img.shields.io/github/stars/yongzhuo/near-synonym?style=social)](https://github.com/yongzhuo/near-synonym/stargazers)
[![Forks](https://img.shields.io/github/forks/yongzhuo/near-synonym.svg?style=social)](https://github.com/yongzhuo/near-synonym/network/members)
[![Join the chat at https://gitter.im/yongzhuo/near-synonym](https://badges.gitter.im/yongzhuo/near-synonym.svg)](https://gitter.im/yongzhuo/near-synonym?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

>>> near-synonym, 中文反义词/近义词/同义词(antonym/synonym)工具包.

# 一、安装
## 1.1 注意事项
   默认不指定numpy版本(标准版numpy==1.20.4)
   标准版本的依赖包详见 requirements-all.txt
   
## 1.2 通过PyPI安装
```
   pip install near-synonym
   使用镜像源, 如：
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple near-synonym
   不带依赖安装, 之后缺什么包再补充什么
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple near-synonym --no-dependencies
```

## 1.3 模型文件
### 版本v0.3.0
 - 新增一种生成反义词/近义词的算法, 构建提示词prompt, 基于BERT-MLM等继续训练, 类似beam_search方法, 生成反义词/近义词;
   ```
   prompt: "xx"的反义词是"[MASK][MASK]"。
   ```
 - 模型权重在[Macropodus/mlm_antonym_model](https://huggingface.co/Macropodus/mlm_antonym_model), 国内镜像[Macropodus/mlm_antonym_model](https://hf-mirror.com/Macropodus/mlm_antonym_model)

### 版本v0.1.0
 - github项目源码自带模型文件只有1w+词向量, 完整模型文件在near_synonym/near_synonym_model, 
 - pip下载pypi包里边没有数据和模型(只有代码), 第一次加载使用huggface_hub下载, 大约为420M;
 - 完整的词向量详见[huggingface](https://huggingface.co/)网站的[Macropodus/near_synonym_model](https://huggingface.co/Macropodus/near_synonym_model), 

### 版本v0.0.3
 - github项目源码自带模型文件只有1w+词向量, 完整模型文件在near_synonym/near_synonym_model, 
 - pip下载的软件包里边只有5w+词向量, 放在data目录下;
 - 完整的词向量详见[huggingface](https://huggingface.co/)网站的[Macropodus/near_synonym_model](https://huggingface.co/Macropodus/near_synonym_model), 
 - 或完整的词向量详见百度网盘分享链接[https://pan.baidu.com/s/1lDSCtpr0r2hKrGrK8ZLlFQ](https://pan.baidu.com/s/1lDSCtpr0r2hKrGrK8ZLlFQ), 密码: ff0y


# 二、使用方式

## 2.1 快速使用方法一, 反义词, 近义词, 相似度
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
[('讨厌', 0.6857), ('厌恶', 0.5406), ('憎恶', 0.485), ('不喜欢', 0.4079), ('冷漠', 0.4051)]
近义词:
[('喜爱', 0.8813), ('爱好', 0.8193), ('感兴趣', 0.7399), ('赞赏', 0.6849), ('倾向', 0.6137)]
"""

w1 = "桂林"
w2 = "柳州"
score = near_synonym.sim(w1, w2)
print(w1, w2, score)
"""
桂林 柳州 0.8947
"""
```


## 2.2 详细使用方法一, 反义词, 相似度
```python3
import near_synonym

word = "喜欢"
word_antonyms = near_synonym.antonyms(word, topk=8, annk=256, annk_cpu=128, batch_size=32,
                     rate_ann=0.4, rate_sim=0.4, rate_len=0.2, rounded=4, is_debug=False)
print("反义词:")
print(word_antonyms)

word1, word2 = "桂林", "柳州"
score = near_synonym.sim(word1, word2, rate_ann=4, rate_sim=4, rate_len=2, 
                                rounded=4, is_debug=False)
print(score)

# 当前版本速度很慢, 召回数量annk_cpu/annk可以调小
```


## 2.3 使用方法二, 基于继续训练 + promt的bert-mlm形式
```python3
import traceback
import os
os.environ["FLAG_MLM_ANTONYM"] = "1"  # 必须先指定

from near_synonym import mlm_synonyms, mlm_antonyms


word = "喜欢"
word_antonyms = mlm_antonyms(word)
word_synonyms = mlm_synonyms(word)
print("反义词:")
print(word_antonyms)
print("近义词:")
print(word_synonyms)

"""
反义词:
[('厌恶', 0.77), ('讨厌', 0.72), ('憎恶', 0.56), ('反恶', 0.49), ('忌恶', 0.48), ('反厌', 0.46), ('厌烦', 0.46), ('反感', 0.45)]
近义词:
[('喜好', 0.75), ('喜爱', 0.64), ('爱好', 0.54), ('倾爱', 0.5), ('爱爱', 0.49), ('喜慕', 0.49), ('向好', 0.48), ('倾向', 0.48)]
"""
```

# 三、技术原理
## 3.1 技术详情
```
near-synonym, 中文反义词/近义词工具包.
流程一(neg_antonym): Word2vec -> ANN -> NLI -> Length

# Word2vec, 词向量, 使用skip-ngram的词向量;
# ANN, 近邻搜索, 使用annoy检索召回;
# NLI, 自然语言推断, 使用Roformer-sim的v2版本, 区分反义词/近义词;
# Length, 惩罚项, 词语的文本长度惩罚;

流程二(mlm_antonym): 构建提示词prompt等重新训练BERT类模型("引号等着重标注, 带句号, 不训练效果很差) -> BERT-MLM(第一个char取topk, 然后从左往右依次beam_search) 
# 构建prompt:
  - "xxx"的反义词是"[MASK][MASK][MASK]"。
  - "xxx"的近义词是"[MASK][MASK][MASK]"。
# 训练MLM
# 一个char一个char地预测, 同beam_search
```

## 3.2 TODO
```
1. 推理加速, 训练小的NLI模型, 替换掉笨重且不太合适的roformer-sim-ft;【20240320已完成ERNIE-SIM，但转为ONNX为340M太大, 考虑浅层网络, 转第四点4.】
2. 使用大模型构建更多的NLI语料;
3. 使用大模型直接生成近义词, 同义词表, 用于前置索引+训练相似度;【20240407已完成】
4. 近义词反义词识别考虑使用经典NLP分类模型, text_cnn/text-rcnn, 基于字向量;【do-ing, 仿transformers写config/tokenizer/model, 方便余预训练模型集成】
5. word2vec召回不太行, 考虑直接使用大模型qwen1.5-0.5b生成;
```

## 3.3 其他实验
```
choice, prompt + bert-mlm;
choice, 如何处理数据/模型文件, 1.huggingface_hub("√")  2.gzip compress whitin 100M in pypi("×");
fail, 使用情感识别, 取得不同情感下的词语(失败, 例如可爱/漂亮同为积极情感);
fail, 使用NLI自然推理, 已有的语料是句子, 不是太适配;
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

# 六、日志
```
2024.10.06, 完成prompt + bert-mlm形式生成反义词/近义词; 
2024.04.14, 修改词向量计算方式(句子级别), 使得句向量的相似度/近义词/反义词更准确一些(依旧很不准, 待改进); 
2024.04.13, 使用huggface_hub下载数据, 即near_synonym_model目录, 在[Macropodus/near_synonym_model](https://huggingface.co/Macropodus/near_synonym_model);
2024.04.07, qwen-7b-chat模型构建28w+词典的近义词/反义词表, 即ci_atmnonym_synonym.json, v0.1.0版本;
2024.03.14, 初始化near-synonym, v0.0.3版本;
```

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

