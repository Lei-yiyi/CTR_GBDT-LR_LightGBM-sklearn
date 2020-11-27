# CTR_GBDT-LR_LightGBM-sklearn

## GBDT（[LightGBM](https://github.com/Microsoft/LightGBM)）
The part of GBDT is proceeded by LightGBM.

## LR（[sklearn](https://github.com/scikit-learn/scikit-learn)）
The part of Logestic Regression is proceeded by sklearn machine learning.

## CTR（[Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)）
The main idea is from the work of Facebook published in 2014 that merging GBDT and LR for CTR prediction.

    GBDT is used for feature transformation while the LR uses the transformed data for prediction

## 步骤
（1）安装 lightgbm 和 sklearn

[lightgbm 安装教程](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#)

[sklearn 安装教程](https://scikit-learn.org/stable/install.html)

（2）下载数据集

[数据集]()

## CTR
FM、W&D这类点击率预估算法 和 CF、矩阵分解、SVD这类topk推荐算法的区别：

（1）推荐系统一般包括 召回->排序（粗排/精排）->重排 三个阶段。

-- 之所以有召回阶段主要是由于算力的限制，以及线上对响应延迟的要求，一般可推荐Item集合可能是百万、千万甚至是上亿级的，对所有Item进行CTR预估不现实。召回阶段即是从所有Item库中先筛选出千级或万级别的Item候选，然后送去排序模型进行打分。

如果排序模型过于复杂或召回Item量过大，往往还会再分为粗排和精排两个阶段，粗排一般是比较简单的线性模型（如LR），精排为较复杂的深度模型，通过粗排进一步缩小候选集到精排模型进行打分。

（2）它们的区别本质是排序和召回的区别，前者负责确定物料的相对顺序，后者负责过滤和甄选物料。

从应用场景上看，前者是推荐排序阶段的核心模块；后者则多半应用在推荐的召回阶段。

从模块目标上看，前者是预估用户点击的概率，一般用曝光点击日志训练模型，曝光未点击做负样本；后者从全库挑选候选，为了覆盖长尾物料一般采用采样剩余未点击物料作为负样本。

从模型输出看，前者是预估每个候选物料的点击概率，是一个概率值；而后者是输出一个topk的列表。










