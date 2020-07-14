import pandas as pd
import numpy as np
import time,datetime
import lightgbm as lgb
from sklearn.metrics import f1_score

# 载入数据
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# 对数据进行排序
train = train.sort_values(['deviceid','guid','ts']) #按值排序
test = test.sort_values(['deviceid','guid','ts'])

# 查看数据是否存在交集
# train deviceid 104736
# test deviceid 56681
# train&test deviceid 46833
# train guid 104333
# test guid 56861
# train&test guid 46654

print('train deviceid',len((set(train['deviceid']))))
print('test deviceid',len((set(test['deviceid']))))
print('train&test deviceid',len((set(train['deviceid'])&set(test['deviceid']))))
print('train guid',len((set(train['guid']))))
print('test guid',len((set(test['guid']))))
print('train&test guid',len((set(train['guid'])&set(test['guid']))))

# 时间格式转化 ts
def time_data2(time_sj):
    data_sj = time.localtime(time_sj/1000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S",data_sj)
    return time_str

#间接地调用函数
train['datetime'] = train['ts'].apply(time_data2)#apply应用函数f
test['datetime'] = test['ts'].apply(time_data2)

train['datetime'] = pd.to_datetime(train['datetime'])#时间转ts，ts转str，再转标准时间。
test['datetime'] = pd.to_datetime(test['datetime'])


# 时间范围，拿到数据第一反应
# 如何继续切分data以完成任务
# 2019-11-07 23:59:59 2019-11-10 23:59:59
# 2019-11-10 23:59:59 2019-11-11 23:59:59
#-------------------------------------------研究数据分布规律
#要和test一起分析，以免故意坑人
print(train['datetime'].min(),train['datetime'].max())
print(test['datetime'].min(),test['datetime'].max())
#转化率
# 7     0.000000
# 8     0.107774
# 9     0.106327
# 10    0.105583

#曝光率
# 7          11
# 8     3674871
# 9     3743690
# 10    3958109
# 11    3653592

#day以后非常重要
train['days'] = train['datetime'].dt.day
test['days'] = test['datetime'].dt.day

train['flag'] = train['days']
test['flag'] = 11

# 8 9 10 11
data = pd.concat([train,test],axis=0,sort=False)
del train,test


# 小时信息
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute


# 缺失值填充
data['guid'] = data['guid'].fillna('abc')#填充字母和填Nan一样，让它有值不为空，xgb和lgb都有缺失值填充的方法
#树对缺失值不算敏感，其他模型要精调。3种方法：
#1忽略，然后熵打折扣。10个样本abc，第10的a缺失，那么算9个的熵*0.9为属性的熵。
#2两边划分。划到两个分支，各占0.5的比率以计算错误率
#3多数投票，平均分数等。如左孩子的数目多熵小就自动划分。
#xgb把缺失值当稀疏矩阵。分裂时不考虑，划分到左右树使loss更小的。预测时，默认右子树。



# 构造历史特征 分别统计前一天 guid deviceid 的相关信息
# 8 9 10 11
history_9 = data[data['days']==8]
history_10 = data[data['days']==9]
history_11 = data[data['days']==10]
history_12 = data[data['days']==11]
del data
#老设备
# 61326
# 64766
# 66547
# 41933
# 42546
print(len(set(history_9['deviceid'])))
print(len(set(history_10['deviceid'])))
print(len(set(history_11['deviceid'])))
print(len(set(history_12['deviceid'])))
print(len(set(history_9['deviceid'])&set(history_10['deviceid'])))
print(len(set(history_10['deviceid'])&set(history_11['deviceid'])))
print(len(set(history_11['deviceid'])&set(history_12['deviceid'])))
#老用户2/3
# 61277
# 64284
# 66286
# 41796
# 42347

print(len(set(history_9['guid'])))
print(len(set(history_10['guid'])))
print(len(set(history_11['guid'])))
print(len(set(history_12['guid'])))
print(len(set(history_9['guid'])&set(history_10['guid'])))
print(len(set(history_10['guid'])&set(history_11['guid'])))
print(len(set(history_11['guid'])&set(history_12['guid'])))
#老新闻也差不多
# 640066
# 631547
# 658787
# 345742
# 350542

print(len(set(history_9['newsid'])))
print(len(set(history_10['newsid'])))
print(len(set(history_11['newsid'])))
print(len(set(history_12['newsid'])))
print(len(set(history_9['newsid'])&set(history_10['newsid'])))
print(len(set(history_10['newsid'])&set(history_11['newsid'])))
print(len(set(history_11['newsid'])&set(history_12['newsid'])))
#为什么要统计？可是还存在1/3的新用户怎么办？基础物理特征。


#------------------------------------------普通特征，feel+try
# netmodel可能会有影响，2D3D
data['netmodel'] = data['netmodel'].map({'o':1, 'w':2, 'g4':4, 'g3':3, 'g2':2})

# pos
data['pos'] = data['pos']



#------------------------------------------构造特征
# deviceid guid timestamp ts 时间特征
# 用户响应时间差
def get_history_visit_time(data1,date2):
    data1 = data1.sort_values(['ts','timestamp'])
    data1['timestamp_ts'] = data1['timestamp'] - data1['ts']
    #target为1是用户点击过,ts是news推，times是用户点
    data1_tmp = data1[data1['target']==1].copy()
    del data1
    for col in ['deviceid','guid']:
        for ts in ['timestamp_ts']:
            f_tmp = data1_tmp.groupby([col],as_index=False)[ts].agg({
                '{}_{}_max'.format(col,ts):'max',
                '{}_{}_mean'.format(col,ts):'mean',
                '{}_{}_min'.format(col,ts):'min',
                '{}_{}_median'.format(col,ts):'median'
            })
        date2 = pd.merge(date2,f_tmp,on=[col],how='left',copy=False)

    return date2

#统计拼接到下一天的特征中
history_10 = get_history_visit_time(history_9,history_10)
history_11 = get_history_visit_time(history_10,history_11)
history_12 = get_history_visit_time(history_11,history_12)

data = pd.concat([history_10,history_11],axis=0,sort=False,ignore_index=True)
data = pd.concat([data,history_12],axis=0,sort=False,ignore_index=True)
del history_9,history_10,history_11,history_12



#------------------------------------------穿越特征
#两种超越特征出于什么考量？CRT的基础手段。
#user画像与item画像。
#点击时间差（反应程度），两次点击时间差（短的时间内物品会有更高的相似度，或者这个时间差会和不同item的长度或类型有关）
#一个穿越特征用户点击时间差，未来特征（无法知道下一次什么时候点击），但是这里跑模型还不错
#先排序了再减上一天即shift-1（按时间进行错位相加减的操作，先groupby再shift）
data = data.sort_values('ts')
data['ts_next'] = data.groupby(['deviceid'])['ts'].shift(-1)
data['ts_next_ts'] = data['ts_next'] - data['ts']


#当前day的一天内的特征
#组合出现的次数
#小时，分钟更加细粒化maybe会更好
for col in [['deviceid'],['guid'],['newsid']]:
    print(col)
    data['{}_days_count'.format('_'.join(col))] = data.groupby(['days'] + col)['id'].transform('count')




print('train and predict')
#9训练，10验证，11预测
X_train = data[data['flag'].isin([9])]
X_valid = data[data['flag'].isin([10])]
X_test = data[data['flag'].isin([11])]



#基础特征
#user.info
#deviceid                128573 non-null object
#guid                    84448 non-null object
#outertag                30268 non-null object
#tag                     63158 non-null object  #或许有类别info
#level                   82654 non-null float64
#personidentification    79644 non-null float64 #有曲线
#followscore             80526 non-null float64 #有曲线
#personalscore           82654 non-null float64
#gender                  55560 non-null float64

#特征细节
#0.0    36427
#2.0     9619
#1.0     9514
#Name: gender, dtype: int64


#ID特征:deviceid，guid，newsid

#标签特征：applist，tag，outertag
#applist{app_1 app_2 app_3 app_86 app_87...}
#tag 都市 言情 outertag 社会热点
#怎么解决NaN？

#时序特征

#app_version, device_vendor, pos


#特征处理：
#历史信息，即前一天的点击量、曝光量、点击率
#前x次曝光、后x次曝光到当前的时间差，后x次到当前曝光的时间差是穿越特征，并且是最强的特征
#二阶交叉特征
#embedding

cate_cols = ['deviceid', 'guid', 'pos', 'app_version',
             'device_vendor', 'netmodel', 'osversion',
             'device_version', 'hour', 'minute', 'second',
             'personalscore', 'gender', 'level_int', 'dist_int',
             'lat_int', 'lng_int', 'gap_before_int', 'ts_before_group',
             'time1', 'gap_after_int', 'ts_after_group',
             'personidentification']
drop_cols = ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
             'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
             'applist_weight', 'tag_key', 'tag_weight', 'outertag_key',
             'outertag_weight', 'newsid']

fillna_cols = ['outertag_len', 'tag_len', 'lng', 'lat','level',
               'followscore', 'dist', 'applist_len', 'ts_before_rank',
              'ts_after_rank']
#data[fillna_cols] = data[fillna_cols].fillna(0)

#-------------------------------------------调库调参
#为什么树深为-1？对于xgb模型测试阶段选重要，判断了过拟合再出手，或者一般先用默认。
#而lgb使用的是num_leaves而不是max_depth来限制模型复杂度。
lgb_param = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1, #3-10，默认6。
    'seed':42,
    'boost_from_average':'false',
    }

#删掉一些当前不需要的无关特征
#新特征：地理位置，net，时分，多个max等，三个count，穿越时间差特征ts_next_ts
feature = [
       'pos','netmodel',  'hour', 'minute',
       'deviceid_timestamp_ts_max', 'deviceid_timestamp_ts_mean',
       'deviceid_timestamp_ts_min', 'deviceid_timestamp_ts_median',
       'guid_timestamp_ts_max', 'guid_timestamp_ts_mean',
       'guid_timestamp_ts_min', 'guid_timestamp_ts_median',
       'deviceid_days_count', 'guid_days_count','newsid_days_count',
        'ts_next_ts'
           ]
target = 'target'


lgb_train = lgb.Dataset(X_train[feature].values, X_train[target].values)
lgb_valid = lgb.Dataset(X_valid[feature].values, X_valid[target].values, reference=lgb_train)
lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=10000, valid_sets=[lgb_train,lgb_valid],
                      early_stopping_rounds=50,verbose_eval=10)
#早停数和日志输出。

p_test = lgb_model.predict(X_valid[feature].values,num_iteration=lgb_model.best_iteration)
xx_score = X_valid[[target]].copy()
xx_score['predict'] = p_test
xx_score = xx_score.sort_values('predict',ascending=False)
xx_score = xx_score.reset_index()#reset_index可以还原索引。这里是排序之后重置索引。

#0.106是对转换率的估计，F1阈值（预测概率大于0.106认为是1会点击，因为转换率也差不多是0.1，10次中1次）
xx_score.loc[xx_score.index<=int(xx_score.shape[0]*0.106),'score'] = 1 #loc提取索引
xx_score['score'] = xx_score['score'].fillna(0)
print(f1_score(xx_score['target'],xx_score['score']))

del lgb_train,lgb_valid
del X_train,X_valid
#0.5129179717875857
#0.5197833317587095
#newsid
# 0.6063125458760602



# f1阈值敏感，所以对阈值做一个简单的迭代搜索。
t0 = 0.05
v = 0.002
best_t = t0
best_f1 = 0
for step in range(201):
    curr_t = t0 + step * v
    y = [1 if x >= curr_t else 0 for x in val_pred]
    curr_f1 = f1_score(val_y, y)
    if curr_f1 > best_f1:
        best_t = curr_t
        best_f1 = curr_f1
        print('step: {}   best threshold: {}   best f1: {}'.format(step, best_t, best_f1))
print('search finish.')

#---------------------------------------------提交
X_train_2 = data[data['flag'].isin([9,10])]


lgb_train_2 = lgb.Dataset(X_train_2[feature].values, X_train_2[target].values)
lgb_model_2 = lgb.train(lgb_param, lgb_train_2, num_boost_round=lgb_model.best_iteration, valid_sets=[lgb_train_2],verbose_eval=10)

p_predict = lgb_model_2.predict(X_test[feature].values)

submit_score = X_test[['id']].copy()
submit_score['predict'] = p_predict
submit_score = submit_score.sort_values('predict',ascending=False)
submit_score = submit_score.reset_index()
submit_score.loc[submit_score.index<=int(submit_score.shape[0]*0.103),'target'] = 1
submit_score['target'] = submit_score['target'].fillna(0)

submit_score = submit_score.sort_values('id')
submit_score['target'] = submit_score['target'].astype(int)

sample = pd.read_csv('./sample.csv')
sample.columns = ['id','non_target']
submit_score = pd.merge(sample,submit_score,on=['id'],how='left')

submit_score[['id','target']].to_csv('./baseline.csv',index=False)




#---------------------------------------------其他：深度学习Method
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM, NFFM, xDeepFM, FGCNN, NFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names

data = pd.read_csv('train.csv')

#稀疏特征，稠密特征
sparse_features = []
dense_features = []

for i in data.columns:
    if data[i].nunique() > 1000:
        dense_features.append(i)
    else:
        if i == "label":
            break
        sparse_features.append(i)

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

#1稀疏特征one-hot，稠密特征minmax
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

#得到维度
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(feature_names)

#所有的特征直接扔
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

#调库调参
# model = NFFM(linear_feature_columns, dnn_feature_columns, l2_reg_dnn=0.01, dnn_dropout=0.5, task='binary')
# model = FGCNN(dnn_feature_columns,l2_reg_dnn=0.01,dnn_dropout=0.5,task='binary')
model = xDeepFM(linear_feature_columns, dnn_feature_columns, l2_reg_dnn=0.01, dnn_dropout=0.5, task='binary')

model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=50000, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

#f1
#0.65




#route
#数据敏感度。
#特征观察度。
#调参炼丹度。
#-------CTR讨论，自我模型搭建。
#补：NN+推荐
