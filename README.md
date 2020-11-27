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

## 推荐系统

推荐系统是个很大的概念，从物理视角来看，具体包括以下模块：

**特征工程**

挖掘user和item特征以及上下文特征，具体包括性别/年龄/类目这些基础属性特征；用户/物品历史数据统计特征；二阶或者高阶交叉组合特征；文本特征、图像特征高级特征。

在对业务场景以及优化目标、算法有充分地理解前提下，才能挖掘出高效特征，该模块涉及技术栈较多，如etl工具/nlp/cv算法等。

其输出内容为召回和排序模块提供数据，除了数据的准确性之外，特征工程pipeline的可靠性与实时性通常对推荐结果影响也较大，这个模块产出特征的质量决定了推荐系统的最终效果上限。

**召回**

从数以百万千万的item候选池中初步筛选出百量级的item，用作推荐排序，降低排序模型的打分压力。

通常会通过规则策略、基于item的协同过滤、基于user的协同过滤等基于统计的传统算法，以上算法原理相对简单，可解释性强，但是如何在大数据上实现向量快速检索是个工程实现的难点。

另外基于机器学习/深度学习的向量召回在业界也越来越普及，如矩阵分解、DSSM等，它们融合了用户和物品双方的特征来做召回，也取得了不错的效果。

**排序**

接收召回模块返回的item，并对其排序后的topK个item作为曝光的重要依据，许多场景排序依据是pctr(预估的点击率)，根据业务场景与目标不同，也可能是cvr/ctcvr等等，这个模块决定了最终曝光给用户的item，这也是学术界和工业界研究比较活跃的一个模块，数据维度规模大且及其稀疏给建模带来了不小的挑战。

排序模型从逻辑回归->因子分解机->深度学习不断进化中。

另外，排序往往分为粗排和精排，粗排模型通常复杂度较低，其作用是扩大精排item候选池。

**线上服务**

该模块用于模型提供线上召回与排序打分服务，通常根据user id/item id从内存数据库拉取user和item特征，拼接后喂给训练好的模型，作出最终打分。拉取数据并将拼接成的数据与模型输入的接口对齐，正确地喂给模型，是这个模块实现的核心逻辑。这个模块的高性能、高并发、低时延是重点关注的点。

影响inference性能的因素，除了线上服务的实现方式外，还与模型本身的复杂度直接相关。当一个系统对实时性要求很高时，特征工程、曝光点击匹配、样本特征拼接这些工作往往也会被搬到线上服务化，对系统的实时性要求越高，此处工程要求越苛刻。

**实验与监控系统**

评价一个算法/策略好坏，除了线下auc/loss等指标，更重要的是线上转化率等业务指标，通过ABtest系统，在小流量上验证线上效果，进而决定是否全量。

如何保证流量均匀且正交，是这个模块的关键，业界一般实现方式为分层hash，保证召回/排序等各层实验互不影响。

另外，netfelix的interleave也是一种新的思路。可靠且稳定的实验系统，为算法/策略的快速升级迭代，提供了基础保障。

## CTR

CTR 预测是很多系统必备部分，比如说搜索系统的排序模块，就包含点击率预测，还有广告系统的点击率预估用于竞价，以及推荐系统中的排序模块，这些都会用到 CTR 模型预估。

建立推荐系统前，首先要根据业务目标确定推荐系统的优化目标，而ctr只是可能被设置成的优化目标之一。
