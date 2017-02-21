
# P2 - 泰坦尼克号生存数据分析

## 项目描述：
### 数据情况
本项目所使用的数据包括泰坦尼克号上 2224 名乘客和船员中 891 名的人口学数据和乘客基本信息。
1. [数据来源](http://t.cn/RIODHCu)
2. [数据集描述](http://t.cn/R2sHf1S)

### 分析目标
希望通过简单的数据分析，解释一下问题：
1. 有哪些因素会让影响乘客的生还率？
2. 这些因素分别会对生还率造成什么样的影响？
3. 这些因素的影响效果如何？

### 猜想
影响乘客生还率的各个可能因素及猜想：

1. 经济地位：社会经济地位越高，越有可能生还。可以从三个变量考虑：Pclass, Fare, Cabin
2. 性别：女性比男性的生还率高。变量：Sex
3. 年龄：年龄越大，生还可能性越小。变量：Age
4. 同伴：同行人数越多，生还可能性越大。变量：SibSp, Parch
5. 港口：不同的港口生还率可能不一样，需要进一步探索。变量: Embarked


### 参考资料

## 1. 数据准备


```python
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
titanic_df = pd.read_csv("titanic-data.csv")
titanic_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    


```python
titanic_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



上表显示了数据集中数值型变量的一般统计量数据。


```python
'Titanic数据集一共有{}条成员记录，有{}名成员生还，总体生还率是{}%'.format(titanic_df['PassengerId'].count(),
                                                titanic_df['Survived'].sum(),
                                                round((100*titanic_df['Survived'].sum())/titanic_df['Survived'].count(),2))
```




    'Titanic数据集一共有891条成员记录，有342名成员生还，总体生还率是38.38%'



### 调查数据中出现的问题
在上文中展示的数据基本信息中，数据集的信息记录相对完整，可以进行初步探索，有现异常情况则可在每项分析前特别解决。在Age一栏只有714条记录，关于这部分的数据调查将在Age变量分析之前进行。

## 2. 变量分析

### 2.1 一维变量分析

#### 2.1.1 生还状况


```python
titanic_df.groupby('Survived')['Survived'].count().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb01cc88>




![png](https://github.com/DaraJin/P2_Investigating_a_Dataset/blob/master/figure/output_13_1.png)


可以看出，生存下来的乘客明显少于未生存下来的乘客。

#### 2.1.2 社会经济地位分布


```python
Pclass_ct=(titanic_df.groupby('Pclass')['PassengerId'].count()) #以Pclass为分组依据，对数据集进行分组。
Pclass_ct.plot(kind='bar') #对分组结果绘图
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb1fcba8>




![png](output_16_1.png)



```python
Pclass_ct
```




    Pclass
    1    216
    2    184
    3    491
    Name: PassengerId, dtype: int64



船上三等类别的乘客人数最多，有491人，二等类别乘客最少，只有184人。

#### 2.1.3 性别分布


```python
sns.factorplot('Sex',data=titanic_df,kind='count')
```




    <seaborn.axisgrid.FacetGrid at 0xb2570b8>




![png](output_20_1.png)


船内男性乘客多于女性乘客。

#### 2.1.4 年龄分布

之前提到有177条乘客年龄记录缺失，我们查看一下这部分记录。


```python
titanic_df[titanic_df['Age'].isnull()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>244373</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2649</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>Emir, Mr. Farred Chehab</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2631</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>1</td>
      <td>3</td>
      <td>O'Dwyer, Miss. Ellen "Nellie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330959</td>
      <td>7.8792</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>0</td>
      <td>3</td>
      <td>Todoroff, Mr. Lalio</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349216</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>Spencer, Mrs. William Augustus (Marie Eugenie)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B78</td>
      <td>C</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>1</td>
      <td>3</td>
      <td>Glynn, Miss. Mary Agatha</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>335677</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>1</td>
      <td>3</td>
      <td>Mamee, Mr. Hanna</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2677</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43</td>
      <td>0</td>
      <td>3</td>
      <td>Kraeff, Mr. Theodor</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349253</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>0</td>
      <td>3</td>
      <td>Rogers, Mr. William John</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>S.C./A.4. 23567</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>0</td>
      <td>3</td>
      <td>Lennon, Mr. Denis</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>370371</td>
      <td>15.5000</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>1</td>
      <td>3</td>
      <td>O'Driscoll, Miss. Bridget</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>14311</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>0</td>
      <td>3</td>
      <td>Samaan, Mr. Youssef</td>
      <td>male</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>2662</td>
      <td>21.6792</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>55</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>Woolner, Mr. Hugh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>19947</td>
      <td>35.5000</td>
      <td>C52</td>
      <td>S</td>
    </tr>
    <tr>
      <th>64</th>
      <td>65</td>
      <td>0</td>
      <td>1</td>
      <td>Stewart, Mr. Albert A</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17605</td>
      <td>27.7208</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>65</th>
      <td>66</td>
      <td>1</td>
      <td>3</td>
      <td>Moubarek, Master. Gerios</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2661</td>
      <td>15.2458</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>76</th>
      <td>77</td>
      <td>0</td>
      <td>3</td>
      <td>Staneff, Mr. Ivan</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349208</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>77</th>
      <td>78</td>
      <td>0</td>
      <td>3</td>
      <td>Moutal, Mr. Rahamin Haim</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>374746</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>82</th>
      <td>83</td>
      <td>1</td>
      <td>3</td>
      <td>McDermott, Miss. Brigdet Delia</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330932</td>
      <td>7.7875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>87</th>
      <td>88</td>
      <td>0</td>
      <td>3</td>
      <td>Slocovski, Mr. Selman Francis</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392086</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>95</th>
      <td>96</td>
      <td>0</td>
      <td>3</td>
      <td>Shorney, Mr. Charles Joseph</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>374910</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>101</th>
      <td>102</td>
      <td>0</td>
      <td>3</td>
      <td>Petroff, Mr. Pastcho ("Pentcho")</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349215</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>107</th>
      <td>108</td>
      <td>1</td>
      <td>3</td>
      <td>Moss, Mr. Albert Johan</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>312991</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>109</th>
      <td>110</td>
      <td>1</td>
      <td>3</td>
      <td>Moran, Miss. Bertha</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>371110</td>
      <td>24.1500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>121</th>
      <td>122</td>
      <td>0</td>
      <td>3</td>
      <td>Moore, Mr. Leonard Charles</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>A4. 54510</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>126</th>
      <td>127</td>
      <td>0</td>
      <td>3</td>
      <td>McMahon, Mr. Martin</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>370372</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>128</th>
      <td>129</td>
      <td>1</td>
      <td>3</td>
      <td>Peter, Miss. Anna</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>F E69</td>
      <td>C</td>
    </tr>
    <tr>
      <th>140</th>
      <td>141</td>
      <td>0</td>
      <td>3</td>
      <td>Boulos, Mrs. Joseph (Sultana)</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2678</td>
      <td>15.2458</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>154</th>
      <td>155</td>
      <td>0</td>
      <td>3</td>
      <td>Olsen, Mr. Ole Martin</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>Fa 265302</td>
      <td>7.3125</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>718</th>
      <td>719</td>
      <td>0</td>
      <td>3</td>
      <td>McEvoy, Mr. Michael</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>36568</td>
      <td>15.5000</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>727</th>
      <td>728</td>
      <td>1</td>
      <td>3</td>
      <td>Mannion, Miss. Margareth</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>36866</td>
      <td>7.7375</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>738</th>
      <td>739</td>
      <td>0</td>
      <td>3</td>
      <td>Ivanoff, Mr. Kanio</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349201</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>739</th>
      <td>740</td>
      <td>0</td>
      <td>3</td>
      <td>Nankoff, Mr. Minko</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349218</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>740</th>
      <td>741</td>
      <td>1</td>
      <td>1</td>
      <td>Hawksford, Mr. Walter James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>16988</td>
      <td>30.0000</td>
      <td>D45</td>
      <td>S</td>
    </tr>
    <tr>
      <th>760</th>
      <td>761</td>
      <td>0</td>
      <td>3</td>
      <td>Garfirth, Mr. John</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>358585</td>
      <td>14.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>766</th>
      <td>767</td>
      <td>0</td>
      <td>1</td>
      <td>Brewe, Dr. Arthur Jackson</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112379</td>
      <td>39.6000</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>768</th>
      <td>769</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. Daniel J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>371110</td>
      <td>24.1500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>773</th>
      <td>774</td>
      <td>0</td>
      <td>3</td>
      <td>Elias, Mr. Dibo</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2674</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>776</th>
      <td>777</td>
      <td>0</td>
      <td>3</td>
      <td>Tobin, Mr. Roger</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>383121</td>
      <td>7.7500</td>
      <td>F38</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>778</th>
      <td>779</td>
      <td>0</td>
      <td>3</td>
      <td>Kilgannon, Mr. Thomas J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>36865</td>
      <td>7.7375</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>783</th>
      <td>784</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Mr. Andrew G</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>790</th>
      <td>791</td>
      <td>0</td>
      <td>3</td>
      <td>Keane, Mr. Andrew "Andy"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>12460</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>792</th>
      <td>793</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Stella Anna</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>793</th>
      <td>794</td>
      <td>0</td>
      <td>1</td>
      <td>Hoyt, Mr. William Fisher</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17600</td>
      <td>30.6958</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0000</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>825</th>
      <td>826</td>
      <td>0</td>
      <td>3</td>
      <td>Flynn, Mr. John</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>368323</td>
      <td>6.9500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>826</th>
      <td>827</td>
      <td>0</td>
      <td>3</td>
      <td>Lam, Mr. Len</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1601</td>
      <td>56.4958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>828</th>
      <td>829</td>
      <td>1</td>
      <td>3</td>
      <td>McCormack, Mr. Thomas Joseph</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>367228</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>832</th>
      <td>833</td>
      <td>0</td>
      <td>3</td>
      <td>Saad, Mr. Amin</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2671</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>837</th>
      <td>838</td>
      <td>0</td>
      <td>3</td>
      <td>Sirota, Mr. Maurice</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>392092</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>839</th>
      <td>840</td>
      <td>1</td>
      <td>1</td>
      <td>Marechal, Mr. Pierre</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>11774</td>
      <td>29.7000</td>
      <td>C47</td>
      <td>C</td>
    </tr>
    <tr>
      <th>846</th>
      <td>847</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. Douglas Bullen</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>849</th>
      <td>850</td>
      <td>1</td>
      <td>1</td>
      <td>Goldenberg, Mrs. Samuel L (Edwiga Grabowska)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>17453</td>
      <td>89.1042</td>
      <td>C92</td>
      <td>C</td>
    </tr>
    <tr>
      <th>859</th>
      <td>860</td>
      <td>0</td>
      <td>3</td>
      <td>Razi, Mr. Raihed</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2629</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>863</th>
      <td>864</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>868</th>
      <td>869</td>
      <td>0</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>345777</td>
      <td>9.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>878</th>
      <td>879</td>
      <td>0</td>
      <td>3</td>
      <td>Laleff, Mr. Kristo</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349217</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 12 columns</p>
</div>



关于这部分缺失的数据，最直接的处理方法是在分析过程中直接去除。而在绘图过程中，去除NAN是自动进行的。值得注意的是，此处需要进一步探索一部分缺失数据的背景，以排除大批具有相似特征的年龄数据缺失的可能性。


```python
titanic_df['Age'].hist(bins=80)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb33da90>




![png](output_26_1.png)


船内乘客年龄在大于10的部分接近正态分布。我们可以计算一些年龄的集中趋势。


```python
print("均值",titanic_df['Age'].mean())
print("中位数",titanic_df['Age'].median())
print("最大值",titanic_df['Age'].max())
print("最小值",titanic_df['Age'].min())
print("标准差",titanic_df['Age'].std())
```

    均值 29.69911764705882
    中位数 28.0
    最大值 80.0
    最小值 0.42
    标准差 14.526497332334044
    


```python
#将船票价格分布
titanic_df['Fare'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xce68438>




![png](output_29_1.png)


票价-数量分布图，低价位的船票数量高度集中，如若要进行分组，按等距分组可能不是一个便于观察的好选择。

### 2.2 二维变量分析

#### 2.2.1 经济地位如何影响生还率？

猜想反应经济地位的因素包括Pclass, Fare, Cabin, Embarked，首先探索这些变量之间的内部关系。


```python
titanic_df.groupby(titanic_df['Embarked'])['PassengerId'].count()
```




    Embarked
    C    168
    Q     77
    S    644
    Name: PassengerId, dtype: int64




```python
P1 = sns.factorplot(x='Embarked',y='Fare',data=titanic_df, order=["C","S","Q"])
P2 = sns.boxplot(x='Embarked', y='Fare', hue='Pclass', data=titanic_df,order=["C","S","Q"])
P2.set(ylim=(0,200))
```




    [(0, 200)]




![png](output_35_1.png)


总体上，C舱的票价水平高于S舱，高于Q舱；一等类别乘客高于二等、三等。


```python
#定义生还率计算函数
def survival_rate(data):
    return data.sum()/data.count()
```


```python
#按Pclass进行分组，提取Survived列，再计算生还比率
Pclass_group=titanic_df.groupby('Pclass')['Survived']
Pclass_group_rate=Pclass_group.apply(survival_rate)
Pclass_group_rate.plot(kind='bar')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
```




    <matplotlib.text.Text at 0x20545c18>




![png](output_38_1.png)



```python
Pclass_gp.head()
```




    Pclass
    1    0.629630
    2    0.472826
    3    0.242363
    Name: Survived, dtype: float64



从上图可观察得，社会经济地位是影响生还率的重要因素。经济地位越高，生还的可能性越高。一等乘客拥有63%的生还率，而三等乘客只有24%左右。


```python
#将船票价格按照quantile分组
Fare_group = titanic_df.groupby(pd.qcut(titanic_df['Fare'],5, precision=0))['Survived']
Fare_group_rate=Fare_group.apply(survival_rate)
Fare_group_rate.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21653780>




![png](output_41_1.png)


从船票价格分布图来看，我们可以尝试按照价格比例分为5组，再计算每一组内的生还率，可以观察得，价格越高，生还率越高。


```python
# 观察舱位对生还率的影响时，首先要对数据进行适当的转化和整理，提取代表舱位类别的首字母
def cabin_class(data):
    return str(data)[0]
titanic_df['Cabin_class']=titanic_df['Cabin'].apply(cabin_class)
cabin_group=titanic_df.groupby(titanic_df['Cabin_class'])['Survived']
cabin_group_rate=cabin_group.apply(survival_rate)
print(cabin_group.count())
print("")
print(cabin_group_rate)
```

    Cabin_class
    A     15
    B     47
    C     59
    D     33
    E     32
    F     13
    G      4
    T      1
    n    687
    Name: Survived, dtype: int64
    
    Cabin_class
    A    0.466667
    B    0.744681
    C    0.593220
    D    0.757576
    E    0.750000
    F    0.615385
    G    0.500000
    T    0.000000
    n    0.299854
    Name: Survived, dtype: float64
    


```python
T舱位似乎是一个异常值。
```


```python
# 寻找异常值的列
a=0
for i in titanic_df['Cabin']:
    if str(i)[0]=='T':
        print(titanic_df.loc[a])
    a=a+1
```


```python
# 排除异常值后的柱状图
cabin_group_rate[["A","B","C","D","E","F","G","n"]].plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2054b358>




![png](output_46_1.png)


有舱位的乘客生还率高于无舱位的乘客，而在各类有舱位的乘客中，B,D,E类舱位生还率最高。

#### 2.2.2 性别对生还率有何影响？


```python
Sex_group=titanic_df.groupby('Sex')['Survived']
Sex_group_rate=Sex_group.apply(survival_rate)
Sex_group_rate.plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
```




    <matplotlib.text.Text at 0x21752a90>




![png](output_49_1.png)



```python
Sex_group_rate
```




    Sex
    female    0.742038
    male      0.188908
    Name: Survived, dtype: float64




```python
Sex_group_rate.plot(kind='bar')
```

性别是影响某位乘客能否生还的重要因素。女性拥有超过74%的生还率，而男性却只有18.9%。

#### 2.2.3年龄对生还率的影响情况如何？


```python
#将NAN部分数据清除
Aged = titanic_df[['Survived', 'Age']].dropna()
#对年龄进行分组
Grouped_Age=Aged.groupby(pd.cut(Aged['Age'],8,labels=["0~10","10~20","20~30","30~40","40~50","50~60","60~70","70~80"]))
#计算分组后的生还率
Age_group=Grouped_Age['Survived']
Age_group_rate = Age_group.apply(survival_rate)
Age_group_rate
```




    Age
    0~10     0.593750
    10~20    0.382609
    20~30    0.365217
    30~40    0.445161
    40~50    0.383721
    50~60    0.404762
    60~70    0.235294
    70~80    0.200000
    Name: Survived, dtype: float64




```python
Age_group_rate.plot()
plt.xlabel("Age") 
plt.ylabel("Survival Rate") 
plt.title("Influence of Age on the Survival Prob")
```




    <matplotlib.text.Text at 0x24c624e0>




![png](output_55_1.png)


整体上来看，生还率随着年龄的增加而下降。

#### 2.2.4 随行旅伴的数量如何影响生还率？


```python
sns.factorplot('SibSp','Survived',data=titanic_df)
sns.factorplot('Parch','Survived',data=titanic_df)
```




    <seaborn.axisgrid.FacetGrid at 0x24848f28>




![png](output_58_1.png)



![png](output_58_2.png)


SibSp和Parch的影响大致相同，将两者统一为Compa变量，意味同伴人数


```python
titanic_df['Compa']=titanic_df['SibSp'] + titanic_df['Parch']
sns.factorplot('Compa','Survived',data=titanic_df)
```




    <seaborn.axisgrid.FacetGrid at 0x245f8320>




![png](output_60_1.png)


观察上图可知，当同伴人数为1~3人时，生存率会高于独身情况，而当同伴人数多于3人时，生还率反而很低。

### 2.3多维变量分析


```python

```


```python
Sex_Pclass_group = titanic_df.groupby(['Sex','Pclass'])['Survived']
Sex_Pclass_group_rate = Sex_Pclass_group.apply(survival_rate)
Sex_Pclass_group_rate
```




    Sex     Pclass
    female  1         0.968085
            2         0.921053
            3         0.500000
    male    1         0.368852
            2         0.157407
            3         0.135447
    Name: Survived, dtype: float64




```python
Sex_Pclass_group_rate.plot(kind='bar')
plt.xlabel("Sex and Passenger Class") 
plt.ylabel("Survival Rate") 
plt.title("Influence of Sex and Class on the Survival Prob")
plt.show()
```


![png](output_65_0.png)



```python
#与上图表现内容相似，一种更简单直观的绘图方式。
sns.factorplot('Pclass','Survived',hue='Sex',data=titanic_df, order=[3,2,1])
```




    <seaborn.axisgrid.FacetGrid at 0x24bce630>




![png](output_66_1.png)


从上图我们可以看出，性别影响大于社会经济地位。即使是第三阶级的女性生还率也还是高于第一阶级男性的生还率。

### 3.结论

实际数据分析结果与猜想基本吻合。

1. 经济地位：经济地位越高，越有可能生还。经济地位可以通过四个因素反应，座位等级、船票价格、是否有舱位以及登船港口。
    a. 座位等级越高，生还率越高；
    b. 船票价格越高，生还率越高；
    c. 有舱位比无舱位的生还率高；
    d. C港登船乘客的生还率高于S港,Q港；
2. 性别：女性比男性的生还率高。
3. 年龄：年龄越大，生还可能性越小。
4. 同伴：同行人数在1~3人时，生还可能性大。

根据以上结论，可以模糊得出一个有高生还率的乘客：一位0~10岁的女性，拥有一等座，随同3人，船票高于40，D舱，在Cherbourg登船。

### 4. 不足
1. 结论中，“0~10岁女性”的群体分类较为不合理，常理来说小孩被救援的可能性较少会受到性别影响。因此在进行人群个人特征分类时，可以考虑将年龄与性别相结合，分为小孩（0~16岁）、女性、男性、老人（60岁以上），这样也许更合乎常理。
2. 177条年龄缺失记录有可能具有某种年龄特征，这种缺失有可能对分析结果造成影响，分析过程尚未排除这种影响存在的可能性。
3. 在分析同伴人数时，在同班人数较高的区间，由于缺乏足够多的数据，结论存在较大的偶然性。
4. 尚未给出各项因素对生还率的影响权重，分析结论依然较为笼统，可以尝试建立模型来精确预测生还率。
5. 在图形可视化上表现单一，需要尝试更多的图形来表现数据特征。
