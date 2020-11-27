# RecommenderSystem-CTR_GBDT-LR_LightGBM-sklearn

## GBDT（[LightGBM](https://github.com/Microsoft/LightGBM)）
The part of GBDT is proceeded by LightGBM.

## LR（[sklearn](https://github.com/scikit-learn/scikit-learn)）
The part of Logestic Regression is proceeded by sklearn machine learning.

## CTR（[Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)）
The main idea is from the work of Facebook published in 2014 that merging GBDT and LR for CTR prediction.

    GBDT is used for feature transformation while the LR uses the transformed data for prediction

## 说明
1）安装 lightgbm 和 sklearn

[lightgbm 安装教程](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#)

[sklearn 安装教程](https://scikit-learn.org/stable/install.html)

2）下载数据集

[数据集]()
