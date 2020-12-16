#!/usr/bin/env python
# coding: utf-8


# 链接：[XGBoost和LightGBM的参数以及调参](https://www.jianshu.com/p/1100e333fcab)

from sklearn.datasets import load_breast_cancer  # breast cancer wisconsin dataset(classification)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression


# ===========================================================================================
# 加载数据集
# ===========================================================================================
print('Load data...')

df = load_breast_cancer()
X = df.data  # (569, 30)
y = df.target  # (569,)

# ===========================================================================================
# 划分训练集和测试集
# ===========================================================================================
X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.3)


'''***************************************** GBDT ****************************************'''
# ===========================================================================================
# 转换为 Dataset 数据格式
# ===========================================================================================
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# ===========================================================================================
# 参数 —— GBDT
# ===========================================================================================
params = {
    'task': 'train', 
    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'metric': {'binary_logloss'}, 
    'num_leaves': 63, 
    'num_trees': 100, 
    'learning_rate': 0.01, 
    'feature_fraction': 0.9, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5, 
    'verbose': 0
}

# ===========================================================================================
# 模型训练 —— GBDT
# ===========================================================================================
print('Start training —— GBDT...')

# train
gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_train)

# ===========================================================================================
# 模型保存 —— GBDT
# ===========================================================================================
print('Save model —— GBDT...')

gbm.save_model('model.txt')

# ===========================================================================================
# 模型预测 —— GBDT（生成特征向量 —— 训练集/测试集）
# ===========================================================================================
# number of leaves, will be used in feature transformation
num_leaf = 63

print('Start predicting —— GBDT（generating feature vectors —— training data）...')

# predict with leaf index of all trees
y_train_pred = gbm.predict(X_train,pred_leaf=True)  # (398, 100)

# feature transformation and write result
print('Writing transformed training data...')
transformed_training_matrix = np.zeros([len(y_train_pred),len(y_train_pred[0]) * num_leaf], 
                                       dtype=np.int64)  # (398, 6300)

for i in range(0,len(y_train_pred)):
    temp_train = np.arange(len(y_train_pred[0])) * num_leaf - 1 + np.array(y_train_pred[i])
    transformed_training_matrix[i][temp_train] += 1

# for i in range(0,len(y_train_pred)):
#     for j in range(0,len(y_train_pred[i])):
#         transformed_training_matrix[i][j * num_leaf + y_train_pred[i][j]-1] = 1



print('Start predicting —— GBDT（generating feature vectors —— testing data）...')

# predict with leaf index of all trees
y_test_pred = gbm.predict(X_test,pred_leaf=True)

# feature transformation and write result
print('Writing transformed testing data...')
transformed_testing_matrix = np.zeros([len(y_test_pred),len(y_test_pred[0]) * num_leaf], 
                                      dtype=np.int64)
for i in range(0,len(y_test_pred)):
    temp_test = np.arange(len(y_test_pred[0])) * num_leaf - 1 + np.array(y_test_pred[i])
    transformed_testing_matrix[i][temp_test] += 1

# for i in range(0,len(y_test_pred)):
#     for j in range(0,len(y_test_pred[i])):
#         transformed_testing_matrix[i][j * num_leaf + y_test_pred[i][j]-1] = 1

# ===========================================================================================
# 特征重要性
# ===========================================================================================
print('Calculate feature importances...')

# feature importances
print('Feature importances:', list(gbm.feature_importance()))
print('Feature importances:', list(gbm.feature_importance("gain")))


'''********************************** Logistic Regression ********************************'''
# ===========================================================================================
# 逻辑回归
# ===========================================================================================
print("LogIstic Regression Start...")

lm = LogisticRegression(penalty='l2',C=0.1) # logestic model construction
lm.fit(transformed_training_matrix,y_train)  # fitting the data

# training data
y_pred_label_train = lm.predict(transformed_training_matrix )
y_pred_est_train = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label
print('number of training data is ' + str(len(y_pred_label_train)))
print(y_pred_est_train)

# testing data
y_pred_label_test = lm.predict(transformed_testing_matrix)
y_pred_est_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
print('number of testing data is ' + str(len(y_pred_label_test)))
print(y_pred_est_test)

# ===========================================================================================
# 模型评估
# ===========================================================================================
print("Start evaluating —— GBDT+LR...")

# calculate predict accuracy —— training data
num_train = 0
for i in range(0,len(y_pred_label_train)):
    if y_train[i] == y_pred_label_train[i]:
        num_train += 1
print("prediction accuracy of training data is " + str((num_train)/len(y_pred_label_train)))

# calculate the Normalized Cross-Entropy —— training data
NE_train = (-1) / len(y_pred_est_train) * sum(((1+y_train)/2 * np.log(y_pred_est_train[:,1]) +  
                                               (1-y_train)/2 * np.log(1 - y_pred_est_train[:,1])))
print("Normalized Cross Entropy of training data is " + str(NE_train))



# calculate predict accuracy —— testing data
num_test = 0
for i in range(0,len(y_pred_label_test)):
    if y_test[i] == y_pred_label_test[i]:
        num_test += 1
print("prediction accuracy of testing data is " + str((num_test)/len(y_pred_label_test)))
    
# calculate the Normalized Cross-Entropy —— testing data
NE_test = (-1) / len(y_pred_est_test) * sum(((1+y_test)/2 * np.log(y_pred_est_test[:,1]) +  
                                               (1-y_test)/2 * np.log(1 - y_pred_est_test[:,1])))
print("Normalized Cross Entropy of testing data is " + str(NE_test))
