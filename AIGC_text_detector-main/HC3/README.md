---
license: CC-BY-SA 4.0
# tags:
# - text-classification
size_scale:
- 10K<n<100K
text:
  text-classification:
    type:
      - binary-class
    language:
      - en
---

# HC3 人类-ChatGPT 问答对比语料集（英文）

## 数据集概述
HC3 语料是 [SimpleAI](https://github.com/Hello-SimpleAI) 团队收集的 对于给定问题的人类-ChatGPT回答文本对。

我们希望：
1. 做出一些开源模型工具来高效检测 ChatGPT 生成的内容；
2. 收集一批有价值的人类和 ChatGPT 对比的中英双语问答语料，来助力相关学术研究。

更多相关信息请查看项目主页 [Hello-SimpleAI/chatgpt-comparison-detection](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) 。

本语料还有中文版本，见 [SimpleAI/HC3-Chinese](https://www.modelscope.cn/datasets/simpleai/HC3-Chinese) .

我们会很快使用 [AdaSeq](https://github.com/modelscope/AdaSeq) (modelscope生态的文本序列理解工具) 实现一行命令训练你自己的检测模型，敬请关注～


### 数据集简介
我们从两种来源的数据构建了本语料:
- 公开的问答数据集：此类数据集通常提供了`问题`和`人类专家回答`，使用问题直接输入给ChatGPT，然后收集`ChatGPT回答`文本。
- wiki百科：我们爬取了 wiki、baidu 百科 的高质量概念 (concept) 和其解释，使用类似 `Please explain what is <concept>`, `"我有一个计算机相关的问题，请用中文回答，什么是 <concept>"` 的形式作为问题，然后收集`ChatGPT回答`文本。其页面上的前几句解释作为`人类专家回答`.

英文版本共5个子数据集：
- `open_qa`
- `reddit_eli5`
- `wiki_csai`
- `medicine`
- `finance`


### 数据集的格式和结构
数据格式采用jsonline格式，有 `question`, `human_answers`, `chatgpt_answers` 三个字段，其中 `question` 为字符串类型，`human_answers` 和 `chatgpt_answers` 都是装有字符串的列表，一般情况下只有一个字符串，也会有多个。

一个具体case的例子如下：

```
{
    "question":"Please explain what is Recommender system",
    "human_answers": [
        "A recommender system, or a recommendation system (sometimes replacing ......",
    ],
    "chatgpt_answers": [
        "A recommender system, or a recommendation system, is a subclass of ",
    ]
}
```

## 数据集版权信息

如果源数据集使用了比 CC-BY-SA 4.0 严格的许可证，我们遵循其原始许可证，否则遵循 CC-BY-SA 4.0 许可证。
详细信息见 [dataset copyright](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection#dataset-copyright).


## 引用方式
```bib
@article{guo-etal-2023-hc3,
    title = "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection",
    author = "Guo, Biyang  and
      Zhang, Xin  and
      Wang, Ziyuan  and
      Jiang, Minqi  and
      Nie, Jinran  and
      Ding, Yuxuan  and
      Yue, Jianwei  and
      Wu, Yupeng",
    journal={arXiv preprint.}
    year = "2023",
}
```
