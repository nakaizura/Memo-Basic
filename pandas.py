import pandas as pd

# 载入数据
train = pd.read_csv('./train.csv')

# 创造数据
data = {'aaa':['1','2','3'],'bbb':['4','5','6']}
datafrm = pd.DataFrame(data)
print(datafrm)

# 或者用pd.DataFrame(dict(zip(labels,cols)))


# 查看数据维度
print(train.shape)

# 载入数据后的格式是什么样子的？
print(type(train)) #DataFrame

# 但是载入后的数据列的格式呢？
print(type(train['ID'])) #Series


# 查看数据的列
train.colums

# 打印前五行
train.iloc[:5,:]
train.head(5)

# 打印后五行，需要注意的是默认从0开始编号
train.iloc[-5,:]
train.tail(5)

# 属性信息
train.info()


# 增加列
train['add0']=0
train['add']=

# 更改名字和计数列表
datafrm.columns=['ccc','ddd']
datafrm.index=['1','2','3']


# 画图
import matplotlib.pyplot as plt
plt.plot(train['Age'])
train['Age'].plot() #可视化某个属性
plt.show()

train.plot() #可视化所有的属性，属性太多了就没啥意义看不清
plt.yscale('log') #有时候可以对y做一个log，能看清一些，可以设置颜色，在AEDA中有
plt.show()

# 存图
fig =train.loc[0:100,['Age','Male']].plot()
fig =fig.get_figure() #取出图
fig.savefig('xxx.jpg')
plt.show()


# Visual Exploratory Data Analysis（EDA）
train.plot(x='Male',y='Age') #线图
plt.show()
train.plot(x='Male',y='Age',kind='scatter')#点图
plt.title('xxxx')
plt.xlable('Male')#设置lable
plt.ylable('Age')
plt.show()
train.plot(y='Age',kind='box') #可以看到outlines点，很有用
plt.show()
train.plot(y='Age',kind='hist') #直方图
plt.show()
train.plot(y='Age',kind='hist',bins=30,range(0,80),rwidth=0.8,normed=True)
plt.show()
train.plot(y='Age',kind='hist',bins=30,range(0,80),rwidth=0.8,cumulative=True,normed=True)#CDF
plt.show()

# Statistical Exploratory Data Analysis（EDA）
train.describe() #看到信息如count，std，min，max，分位数等等
train['Age'].count() #不计空
train['Age'].mean()
train.quantile(0.5)
train.quantile([0.5,0.75])

train['Age'].describe()
train['Age'].unique()

indices=train['Age']=='20' #拿到某些值对应的index
Age20=train.loc[indices,:]
Age20['Age'].unique()
Age20.head(2) #可以看到此时的行数和原来是对应的，并不是连续的


# Additional Visual Exploratory Data Analysis（EDA）
train.plot(kind='hist',bins=80,rwidth=0.8,alpha=0.3,range=(0,80))#全属性，用颜色控制
plt.show()
Age20.plot(kind='hist',bins=80,rwidth=0.8,alpha=0.3,range=(0,80))
plt.show()





# 时间序列
test = pd.read_csv('./test.csv',parse_dates=True, index_col='Date')#把时间当标识
test = test.drop(columns='Unnamed: 0')#去掉默认的标识

test.loc['2019-08-01 08:00:00','Age'] #定位
test.loc['2019']
test.loc['2019-Oct-01'] #用了标准的时间，用oct这种也是可以的

time = pd.to_datatime(['2019-08-01 08:00','2019-08-01 20:00'])#string转成标准格式
time
test.reindex(time) #用变了之后的格式重新排列行，如果time有test没有的会NaN很方便
test.reindex(time,method='ffill') #跟上一条一样，NaN的填补
test.reindex(time,method='ffill') #跟下一条一样

# 时间resample，重新改变数据的粒度
daily_mean=test.resample('D').mean #按day，如果没有改天就NaN
daily_mean=test.resample('M').mean #按month
daily_mean=test.resample('6W').mean #每6周


# 其他
test['name'].str.upper()
test['name'].str.contains('i')#包含i
test['Date'].dt.hour #得到时

test['Date'].dt.tz_localize('US/Eastern') #美东时间
test['Date'].dt.tz_localize('US/Eastern')dt.tz_convert('Europe/Paris')#在美东相对巴黎

test.loc['2019',['Units','Price']].plot(subplots=True) #画两个图分子图的快速写法


