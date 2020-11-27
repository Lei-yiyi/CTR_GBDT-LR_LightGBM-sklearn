# CTR_GBDT-LR_LightGBM-sklearn

## GBDT（[LightGBM](https://github.com/Microsoft/LightGBM)）
The part of GBDT is proceeded by LightGBM.

## LR（[sklearn](https://github.com/scikit-learn/scikit-learn)）
The part of Logestic Regression is proceeded by sklearn machine learning.

## CTR（[Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)）
The main idea is from the work of Facebook published in 2014 that merging GBDT and LR for CTR prediction.

    GBDT is used for feature transformation while the LR uses the transformed data for prediction

## 步骤
1）安装 lightgbm 和 sklearn

[lightgbm 安装教程](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#)

[sklearn 安装教程](https://scikit-learn.org/stable/install.html)

2）下载数据集

[数据集]()

## CTR
-推荐系统一般包括 召回->排序->重排 三个阶段，CTR预估就是排序阶段的主要任务，而一般常说的推荐算法如协同过滤、矩阵分解等都是对应到召回阶段的策略。

之所以有召回阶段主要是由于算力的限制，以及线上对响应延迟的要求，一般可推荐Item集合可能是百万、千万甚至是上亿级的，对所有Item进行CTR预估不现实。召回阶段即是从所有Item库中先筛选出千级或万级别的Item候选，然后送去排序模型进行打分，如果排序模型过于复杂或召回Item量过大，往往还会再分为粗排和精排两个阶段，粗排一般是比较简单的线性模型（如LR），精排为较复杂的深度模型，通过粗排进一步缩小候选集到精排模型进行打分。

