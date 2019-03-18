# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:17:16 2018

@author: yaowu
"""
#==============================================================================
# 导入模块 ： 导入所需要的模块
# 数据清洗 ： 在研究变量的基础上进行数据清洗
# 变量筛选 ：
# 建立模型 ： 完成数据建模
# 评估模型 ： 最终评估模型质量
#==============================================================================
## 1.1 导入模块
# 导入一些数据分析和数据挖掘常用的包
import numpy as np,pandas as pd,os,seaborn as sns,matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
#statsmodels,统计模型，计量经济学是一个包含统计模型、统计测试和统计数据挖掘python模块
#outliers_influence，
#variance_inflation_factor，方差膨胀因子(Variance Inflation Factor,VIF):是指解释变量之间存在多重共线性时的方差与不存在多重共线性时的方差之比
from sklearn.preprocessing import StandardScaler
#sklearn.preprocessing,数据预处理模块
#StandardScaler,去均值和方差归一化模块
from sklearn.decomposition import PCA
#分解，降维
## 1.2 研究变量的情况 导入包之后加载一下数据集，查看一下数据的基础情况，再考虑下一步的处理
# 加载一下数据，并打印部分数据，查看一下数据的情况
os.chdir(r'C:\Users\A3\Desktop\2：项目\项目\项目19：基于随机森林算法的房屋价格预测模型\房屋价格预测')
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
print(data_train.head())
print(data_test.head())
# 查看数据的列名和每列的数据格式，方便后面对数据进行处理
#data_train.columns
#data_train.info
#查看数据结构
data_train_dtypes = data_train.dtypes
#print(data_train_dtypes)
'''
#==============================================================================
# 对因变量进行具体情况具体分析，主要查看因变量的统计情况，包含偏度和峰度等。
# 峰度：峰度（Kurtosis）是描述某变量所有取值分布形态陡缓程度的统计量。
# 它是和正态分布相比较的。
# Kurtosis=0 与正态分布的陡缓程度相同。
# Kurtosis>0 比正态分布的高峰更加陡峭——尖顶峰
# Kurtosis<0 比正态分布的高峰来得平稳——平顶峰     
# 计算公式：β = M_4 /σ^4 
# 偏度：偏度（Skewness）是描述某变量取值分布对称性的统计量。
# Skewness=0 分布形态与正态分布偏度相同
# Skewness>0 正偏差数值较大，为正偏或右偏。长尾巴拖在右边。
# Skewness<0 负偏差数值较大，为负偏或左偏。长尾巴拖在左边。 
# 计算公式： S= (X^ - M_0)/δ Skewness 越大，分布形态偏移程度越大。
#==============================================================================
'''
# 查看因变量价格的情况，进行基础分析
sns.distplot(data_train['SalePrice'])
plt.show()
# 对房屋价格金额的数值图形化，查看一下
sns.set(style="darkgrid")
titanic = pd.DataFrame(data_train['SalePrice'].value_counts())
titanic.columns = ['SalePrice_count']
ax = sns.countplot(x="SalePrice_count", data=titanic)
plt.show()

# 从房屋价格的正态分布情况，查看房屋价格的峰度和偏度情况
print('房屋价格偏度：%f' % (data_train['SalePrice'].skew()))
print('房屋价格峰度：%f' % (data_train['SalePrice'].kurt()))
'''
#==============================================================================
# 分析数据中的缺失值情况，如果超过阈值15%，则删除这个变量，其他变量根据类别或者是数值型变量进行填充。
#   具体得到的情况如下：
# 
# 有缺失的对应的变量名称 
# * PoolQC Pool quality 游泳池质量 
# * MiscFeature Miscellaneous feature not covered in other categories 其他杂项，例如网球场、第二车库等 
# * Alley Type of alley access to property 胡同小路，是碎石铺就的还是其他等等 
# * Fence Fence quality 护栏质量 
# * FireplaceQu Fireplace quality 壁炉的质量 
# * LotFrontage Linear feet of street connected to property 街道情况 
# * GarageFinish Interior finish of the garage 车库完成情况 
# * GarageQual Garage quality 车库质量 
# * GarageType Garage location 车库的位置 
# * GarageYrBlt Year garage was built 车库的建筑年龄 
# * GarageCond Garage condition 车库的条件 
# * BsmtExposure Refers to walkout or garden level walls 花园的墙壁情况 
# * BsmtFinType2 Rating of basement finished area (if multiple types) 地下室的完工面积 
# * BsmtQual Evaluates the height of the basement 地下室的高度 
# * BsmtCond Evaluates the general condition of the basement 地下室的质量情况 
# * BsmtFinType1 Rating of basement finished area 地下室完工面积 
# * MasVnrType Masonry veneer type 表层砌体类型 
# * MasVnrArea Masonry veneer area in square feet 砖石镶面面积平方英尺 
# * Electrical Electrical system 电气系统
# 
#==============================================================================
'''
# 在进行图形分析之前，先分析一下数据中缺失值的情况
miss_data = data_train.isnull().sum().sort_values(ascending=False)  # 缺失值数量
total = data_train.isnull().count()  # 总数量
miss_data_tmp = (miss_data / total).sort_values(ascending=False)  # 缺失值占比
# 添加百分号
def precent(X):
    X = '%.2f%%' % (X * 100)
    return X
miss_precent = miss_data_tmp.map(precent)
# 根据缺失值占比倒序排序
miss_data_precent = pd.concat([total, miss_precent, miss_data_tmp], axis=1, keys=[
                              'total', 'Percent', 'Percent_tmp']).sort_values(by='Percent_tmp', ascending=False)
# 有缺失值的变量打印出来
print(miss_data_precent[miss_data_precent['Percent'] != '0.00%'])

#* 将缺失值比例大于15%的数据全部删除，剩余数值型变量用众数填充、类别型变量用None填充。*

drop_columns = miss_data_precent[miss_data_precent['Percent_tmp'] > 0.15].index
data_train = data_train.drop(drop_columns, axis=1)
data_test = data_test.drop(drop_columns, axis=1)
# 类别型变量
class_variable = [
    col for col in data_train.columns if data_train[col].dtypes == 'O']
# 数值型变量
numerical_variable = [
    col for col in data_train.columns if data_train[col].dtypes != 'O']  # 大写o
print('类别型变量:%s' % class_variable, '数值型变量:%s' % numerical_variable)

# 数值型变量用中位数填充，test集中最后一列为预测价格，所以不可以填充
from sklearn.preprocessing import Imputer
#Imputer填充模块
padding = Imputer(strategy='median')
data_train[numerical_variable] = padding.fit_transform(
    data_train[numerical_variable])
data_test[numerical_variable[:-1]
          ] = padding.fit_transform(data_test[numerical_variable[:-1]])
# 类别变量用None填充
data_train[class_variable] = data_train[class_variable].fillna('None')
data_test[class_variable] = data_test[class_variable].fillna('None')

# 对所有的数据图形化，主要判别一下数据和因变量之间的关联性，后期我们再采用方差验证来选择变量。
# 用图形化只是为了大致查看数据情况，构建简单模型使用。对变量的选择并不准确，对于较多变量，用图形先判断一下，也是一个办法；
# 如果变量较多，这种办法效率低，并且准确度也差，下面随机选择了几个变量画图看一下状况，后期并不使用这种办法来选择变量

# 根据变量，选择相关变量查看与因变量之间的关系
# CentralAir 中央空调
data = pd.concat([data_train['SalePrice'], data_train['CentralAir']], axis=1)
fig = sns.boxplot(x='CentralAir', y="SalePrice", data=data)
plt.title('CentralAir')
plt.show()
# 有中央空调的房价更高

# MSSubClass 房屋类型
data = pd.concat([data_train['SalePrice'], data_train['MSSubClass']], axis=1)
fig = sns.boxplot(x='MSSubClass', y="SalePrice", data=data)
plt.title('MSSubClass')
plt.show()
# 房屋类型方面和房屋价格之间关联度不大

# MSZoning 房屋区域，例如有高密度住宅、低密度住宅，猜想这个变量和房屋价格之间的关系比较密切
data = pd.concat([data_train['SalePrice'], data_train['MSZoning']], axis=1)
fig = sns.boxplot(x='MSZoning', y="SalePrice", data=data)
plt.title('MSZoning')
plt.show()
# 实际结果显示和房屋价格的关系不大


# LotArea 这个变量猜想会与房屋价格有直接的关系
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x=data_train['SalePrice'], y=data_train['LotArea'])
plt.xlabel('SalePrice')
plt.ylabel('LotArea')
plt.title('LotArea')
plt.show()
# 地块面积与房屋价格有较高相关性


# Street
data = pd.concat([data_train['SalePrice'], data_train['Street']], axis=1)
fig = sns.boxplot(x='Street', y="SalePrice", data=data)
plt.title('Street')
plt.show()
# 街道类型和房屋价格有较高相关性


# LotShape
data = pd.concat([data_train['SalePrice'], data_train['LotShape']], axis=1)
fig = sns.boxplot(x='LotShape', y="SalePrice", data=data)
plt.title('LotShape')
plt.show()
# 房屋形状和房屋价格相关不明显

# LandContour
data = pd.concat([data_train['SalePrice'], data_train['LandContour']], axis=1)
fig = sns.boxplot(x='LandContour', y="SalePrice", data=data)
plt.title('LandContour')
plt.show()
# 房屋所在地是否平整和房屋价格关系较弱

# Utilities
data = pd.concat([data_train['SalePrice'], data_train['Utilities']], axis=1)
fig = sns.boxplot(x='Utilities', y="SalePrice", data=data)
plt.title('Utilities')
plt.show()
# 公共设施和房屋价格基本无关系

# LotConfig
data = pd.concat([data_train['SalePrice'], data_train['LotConfig']], axis=1)
fig = sns.boxplot(x='LotConfig', y="SalePrice", data=data)
plt.title('LotConfig')
plt.show()
'''
#==============================================================================
# 因为变量较多，直接采用关系矩阵，查看各个变量和因变量之间的关系，使用的时候采用spearman系数，原因: 
# Pearson 线性相关系数只是许多可能中的一种情况，为了使用Pearson 线性相关系数必须假设数 
# 据是成对地从正态分布中取得的，并且数据至少在逻辑范畴内必须是等间距的数据。如果这两条件 
# 不符合，一种可能就是采用Spearman 秩相关系数来代替Pearson 线性相关系数。Spearman 秩相关系 
# 数是一个非参数性质（与分布无关）的秩统计参数，由Spearman 在1904 年提出，用来度量两个变 
# 量之间联系的强弱(Lehmann and D’Abrera 1998)。Spearman 秩相关系数可以用于R 检验，同样可以 
# 在数据的分布使得Pearson 线性相关系数不能用来描述或是用来描述或导致错误的结论时，作为变 
# 量之间单调联系强弱的度量。
# Spearman对原始变量的分布不作要求，属于非参数统计方法，适用范围要广些。
# 理论上不论两个变量的总体分布形态、样本容量的大小如何，都可以用斯皮尔曼等级相关来进行研究 。
#==============================================================================
'''
#==============================================================================
# 变量处理 ：
#
# 在变量处理期间，我们先考虑处理更简单的数值型变量，再考虑处理复杂的类别型变量；
#
# 其中数值型变量，需要先考虑和因变量的相关性，其次考虑变量两两之间的相关性，再考虑变量的多重共线性；
#
# 类别型变量除了考虑相关性之外，需要进行编码。
#==============================================================================
# 绘制热力图，查看一下数值型变量之间的关系
corrmat = data_train[numerical_variable].corr('spearman')
f, ax = plt.subplots(figsize=(12, 9))
ax.set_xticklabels(corrmat, rotation='horizontal')
sns.heatmap(np.fabs(corrmat), square=False, center=1)
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360)
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=90)
plt.show()

# 计算变量之间的相关性
numerical_variable_corr = data_train[numerical_variable].corr('spearman')
numerical_corr = numerical_variable_corr[
    numerical_variable_corr['SalePrice'] > 0.5]['SalePrice']
print(numerical_corr.sort_values(ascending=False))
index0 = numerical_corr.sort_values(ascending=False).index
# 结合考虑两两变量之间的相关性
print(data_train[index0].corr('spearman'))

#==============================================================================
# 结合上述情况，选择出如下的变量：
# Variable	相关性
# OverallQual	0.809829
# GrLivArea	0.731310
# GarageCars	0.690711
# YearBuilt	0.652682
# FullBath	    0.635957
# TotalBsmtSF	0.602725
# YearRemodAdd	0.571159
# Fireplaces	0.519247
#==============================================================================

#在这个基础上再考虑变量之间的多重共线性
new_numerical = ['OverallQual', 'GrLivArea', 'GarageCars',
                 'YearBuilt', 'FullBath', 'TotalBsmtSF', 'YearRemodAdd', 'Fireplaces']
X = np.matrix(data_train[new_numerical])
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
VIF_list
#可以明显看到数据有很强的多重共线性，对数据进行标准化和降维

Scaler = StandardScaler()
data_train_numerical = Scaler.fit_transform(data_train[new_numerical])
pca = PCA(n_components=7)
newData_train = pca.fit_transform(data_train_numerical)
newData_train

Scaler = StandardScaler()
data_test_numerical = Scaler.fit_transform(data_test[new_numerical])
pca = PCA(n_components=7)
newData_test = pca.fit_transform(data_test_numerical)
newData_test

newData_train = pd.DataFrame(newData_train)
# newData
y = np.matrix(newData_train)
VIF_list = [variance_inflation_factor(y, i) for i in range(y.shape[1])]
print(newData_train, VIF_list)

#从上面的数据标准化和降维之后，已经消除了多重共线性了。接下来处理类别数据
# 单因素方差分析

from statsmodels.formula.api import ols
#单因素方差分析模块
from statsmodels.stats.anova import anova_lm
#双因素方差分析模块
a = '+'.join(class_variable)
formula = 'SalePrice~ %s' % a
anova_results = anova_lm(ols(formula, data_train).fit())
print(anova_results.sort_values(by='PR(>F)'))
#我们需要看的是单个自变量对因变量SalePrice的影响，因此这里使用单因素方差分析。
#分析结果中 P_values（PR(>F)）越小，说明该变量对目标变量的影响越大。通常我们只选择 P_values 小于 0.05 的变量
# 从变量列表和数据中剔除 P 值大于 0.05 的变量
del_var = list(anova_results[anova_results['PR(>F)'] > 0.05].index)
del_var
# 移除变量
for each in del_var:
    class_variable.remove(each)
# 移除变量数据
data_train = data_train.drop(del_var, axis=1)
data_test = data_test.drop(del_var, axis=1)
# 对类别型变量进行编码
def factor_encode(data):
    map_dict = {}
    for each in data.columns[:-1]:
        piv = pd.pivot_table(data, values='SalePrice',
                             index=each, aggfunc='mean')
        piv = piv.sort_values(by='SalePrice')
        piv['rank'] = np.arange(1, piv.shape[0] + 1)
        map_dict[each] = piv['rank'].to_dict()
    return map_dict

# 调用上面的函数，对名义特征进行编码转换
class_variable.append('SalePrice')
map_dict = factor_encode(data_train[class_variable])
for each_fea in class_variable[:-1]:
    data_train[each_fea] = data_train[each_fea].replace(map_dict[each_fea])
    data_test[each_fea] = data_test[each_fea].replace(map_dict[each_fea])
#因为上面已经完成编码，这里我们再根据相关性判断和选择变量
class_coding_corr=data_train[class_variable].corr('spearman')['SalePrice'].sort_values(ascending=False)
print(class_coding_corr[class_coding_corr>0.5])
class_0=class_coding_corr[class_coding_corr>0.5].index
data_train[class_0].corr('spearman')
#==============================================================================
# SalePrice	Neighborhood	ExterQual	BsmtQual	KitchenQual	GarageFinish	GarageType	Foundation
# SalePrice	1.000000	0.755779	0.684014	0.678026	0.672849	0.633974	0.598814	0.573580
# Neighborhood	0.755779	1.000000	0.641588	0.650639	0.576106	0.542172	0.520204	0.584784
# ExterQual	0.684014	0.641588	1.000000	0.645766	0.725266	0.536103	0.444759	0.609009
# BsmtQual	   0.678026	0.650639	0.645766	1.000000	0.575112	0.555535	0.468710	0.669723
# KitchenQual	0.672849	0.576106	0.725266	0.575112	1.000000	0.480438	0.412784	0.546736
# GarageFinish	0.633974	0.542172	0.536103	0.555535	0.480438	1.000000	0.663870	0.516078
# GarageType	0.598814	0.520204	0.444759	0.468710	0.412784	0.663870	1.000000	0.445793
# Foundation	0.573580	0.584784	0.609009	0.669723	0.546736	0.516078	0.445793	1.000000
#==============================================================================
#查找两两之间的共线性之后，我们保留如下变量Neighborhood，ExterQual，BsmtQual，GarageFinish，GarageType，GarageType；
#接下来尝试查看多重共线性
class_variable = ['Neighborhood', 'ExterQual', 'BsmtQual',
                 'GarageFinish', 'GarageType', 'Foundation']
X = np.matrix(data_train[class_variable])
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
VIF_list
#==============================================================================
# [9.613821793463067,
#  31.39188664149662,
#  33.53637481741086,
#  22.788724064203514,
#  18.362389057630754,
#  15.022566626297733]
#==============================================================================
Scaler = StandardScaler()
data_train_class = Scaler.fit_transform(data_train[class_variable])
pca = PCA(n_components=3)
newData_train_class = pca.fit_transform(data_train_class)
newData_train_class
#==============================================================================
# array([[-1.77187082,  0.39144961, -0.00666812],
#        [-0.56662675, -0.71029462, -0.51616971],
#        [-1.77187082,  0.39144961, -0.00666812],
#        ...,
#        [-1.53107491,  0.09189354, -1.97872919],
#        [ 1.08133883, -0.73280347, -0.2622237 ],
#        [-0.15886914, -1.39847287, -0.0633631 ]])
#==============================================================================
Scaler = StandardScaler()
data_test_class = Scaler.fit_transform(data_test[class_variable])
pca = PCA(n_components=3)
newData_test_class = pca.fit_transform(data_test_class)
newData_test_class
#==============================================================================
# array([[ 1.02960083, -0.69269663,  0.29882836],
#        [ 1.02960083, -0.69269663,  0.29882836],
#        [-1.36691619, -0.77507848, -1.26800226],
#        ...,
#        [ 1.62092967,  0.49991231,  0.28578798],
#        [ 1.25970992,  2.79139405, -1.00467437],
#        [-1.16601113, -0.81964858, -1.47794992]])
#==============================================================================
newData_train_class = pd.DataFrame(newData_train_class)
y = np.matrix(newData_train_class)
VIF_list = [variance_inflation_factor(y, i) for i in range(y.shape[1])]
print(VIF_list)
#==============================================================================
# [1.0, 0.9999999999999993, 0.9999999999999996]
#==============================================================================
#训练集
newData_train_class = pd.DataFrame(newData_train_class)
newData_train_class.columns = ['降维后类别A','降维后类别B','降维后类别C']
newData_train = pd.DataFrame(newData_train)
newData_train.columns = ['降维后数值A','降维后数值B','降维后数值C','降维后数值D','降维后数值E','降维后数值F','降维后数值G']
target = data_train['SalePrice']
target = pd.DataFrame(target)
train = pd.concat([newData_train_class,newData_train],axis=1, ignore_index=True)

#测试集
newData_test_class = pd.DataFrame(newData_test_class)
newData_test_class.columns = ['降维后类别A','降维后类别B','降维后类别C']
newData_test = pd.DataFrame(newData_test)
newData_test.columns = ['降维后数值A','降维后数值B','降维后数值C','降维后数值D','降维后数值E','降维后数值F','降维后数值G']
test = pd.concat([newData_test_class,newData_test],axis=1, ignore_index=True)

from sklearn.model_selection import train_test_split
#train_test_split函数用于将矩阵随机划分为训练子集和测试子集,并返回划分好的训练集测试集样本和训练集测试集标签
from sklearn.ensemble import RandomForestRegressor
#随机森林
from sklearn.linear_model import LogisticRegression
#逻辑回归
from sklearn import svm
#支持向量机
train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)
# 当前参数为默认参数
m = RandomForestRegressor()
m.fit(train_data, train_target)
from sklearn.metrics import r2_score
score = r2_score(test_target,m.predict(test_data))
print(score)
#==============================================================================
# 0.8407489786565608
# D:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
#   after removing the cwd from sys.path.
#==============================================================================
lr = LogisticRegression(C=1000.0,random_state=0)
lr.fit(train_data, train_target)
from sklearn.metrics import r2_score
score = r2_score(test_target,lr.predict(test_data))
print(score)
#==============================================================================
# D:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
#   y = column_or_1d(y, warn=True)
# 0.6380328421527758
#==============================================================================
clf = svm.SVC(kernel = 'poly')
clf.fit(train_data, train_target)
score = r2_score(test_target,clf.predict(test_data))
print(score)
#==============================================================================
# D:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
#   y = column_or_1d(y, warn=True)
# 0.775319597309544
#==============================================================================
#==============================================================================
# 结论就是逻辑回归等模型性能比较差，即使已经经过了正则化、PCA降维、去除多重共线性等。kaggle的比赛起手就应该用XGB。
# 下面尝试使用一下网格搜索的方式看能否提高一下随机森林的性能。
#==============================================================================
from sklearn.grid_search import GridSearchCV
#网格搜索功能
from sklearn.pipeline import Pipeline

#==============================================================================
# D:\ProgramData\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
#   "This module will be removed in 0.20.", DeprecationWarning)
# D:\ProgramData\Anaconda3\lib\site-packages\sklearn\grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
#   DeprecationWarning)
#==============================================================================
param_grid = {'n_estimators':[1,10,100,200,300,400,500,600,700,800,900,1000,1200],'max_features':('auto','sqrt','log2')}
m = GridSearchCV(RandomForestRegressor(),param_grid)
m=m.fit(train_data,train_target.values.ravel())
print(m.best_score_)
print(m.best_params_)
#==============================================================================
# 0.8642193810202421
# {'max_features': 'sqrt', 'n_estimators': 500}
#==============================================================================
#通过网格搜索找到最佳的参数后，代入模型，模型完成
m = RandomForestRegressor(n_estimators=200,max_features='sqrt')
m.fit(train_data, train_target.values.ravel())
predict = m.predict(test)
test = pd.read_csv('test.csv')['Id']
sub = pd.DataFrame()
sub['Id'] = test
sub['SalePrice'] = pd.Series(predict)
sub.to_csv('Predictions.csv', index=False)

print('finished!')

---------------------
# 作者：孙耀武
# 来源：CSDN
# 原文：https://blog.csdn.net/sunyaowu315/article/details/82982989
# 版权声明：本文为博主原创文章，转载请附上博文链接！