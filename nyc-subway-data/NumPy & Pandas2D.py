# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 23:12:13 2017

@author: 呵呵
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



"""

#-----------------------------------------------------------------------------#
UNIT Remote unit that collects turnstile information. Can collect from multiple banks of turnstiles.Large subway stations can have more than one unit.

DATEn Date in “yyyymmdd”(20110521)format.

TIMEn Time in “hh:mm:ss” (08:05:02) format.

ENTRIESn Raw reading of cummulative turnstile entries from the remote unit. Occasionally resets to 0.

EXITSn Raw reading of cummulative turnstile exits from the remote unit. Occasionally resets to 0.

ENTRIESn_hourly Difference in ENTRIES from the previous REGULAR reading.

EXITSn_hourly Difference in EXITS from the previous REGULAR reading.

datetime Date and time in “yyyymmdd hh:mm:ss” format (20110501 00:00:00). Can be parsed into a Pandas datetime object without modifications.

hour Hour of the timestamp from TIMEn. Truncated rather than rounded.

day_week Integer (0 6Mon Sun)corresponding to the day of the week.

weekday Indicator (0 or 1) if the date is a weekday (Mon Fri).

station Subway station corresponding to the remote unit.

latitude Latitude of the subway station corresponding to the remote unit.

longitude Longitude of the subway station corresponding to the remote unit.

conds Categorical variable of the weather conditions (Clear, Cloudy etc.) for the time and location.

fog Indicator (0 or 1) if there was fog at the time and location.

precipi Precipitation in inches at the time and location.

pressurei Barometric pressure in inches Hg at the time and location.

rain Indicator (0 or 1) if rain occurred within the calendar day at the location.

tempi Temperature in ℉ at the time and location.

wspdi Wind speed in mph at the time and location.

meanprecipi Daily average of precipi for the location.

meanpressurei Daily average of pressurei for the location.

meantempi Daily average of tempi for the location.

meanwspdi Daily average of wspdi for the location.

weather_lat Latitude of the weather station the weather data is from.

weather_lon Longitude of the weather station the weather data is from.
#-----------------------------------------------------------------------------#

"""




#提出问题
#-----------------------------------------------------------------------------#
#我提出的问题

#Q1:每个车站日均人流量
#Q2:节假日每个车站的人流量均值是否比平常高
#Q3:降雨日人流量与平时人流量对比
#Q4:不同天气下的平均人流量对比
#Q5:每日5个时间段内平均人流量

#课程问题

# 有哪些变量与地铁客流量高低相关

# Q1 哪个车站有最多的人流量
# Q2 人流量高峰期在哪个时间段
# Q3 天气对客流量的影响是什么

# 天气变化的模式有什么特点

# Q4 5月的气温是否稳步升高
# Q5 纽约各地区的天气有何不同

#-----------------------------------------------------------------------------#



#练习：二维NumPy数组
#-----------------------------------------------------------------------------#
"""
# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

# Change False to True for each block of code to see what it does

# Accessing elements
if False:
    print (ridership[1, 3])
    print (ridership[1:3, 3:5])
    print (ridership[1, :])
    
# Vectorized operations on rows or columns
if False:
    print (ridership[0, :] + ridership[1, :])
    print (ridership[:, 0] + ridership[:, 1])
    
# Vectorized operations on entire arrays
if False:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    print (a + b)



def mean_riders_for_max_station(ridership):
    '''
    Fill in this function to find the station with the maximum riders on the
    first day, then return the mean riders per day for that station. Also
    return the mean ridership overall for comparsion.
    
    Hint: NumPy's argmax() function might be useful:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
    '''
    
    x = ridership[0,:].argmax()
    overall_mean = ridership.mean() # ReNoneplace this with your code
    mean_for_max = ridership[:,x].mean() # Replace this with your code
    
    return (overall_mean, mean_for_max)
"""
#-----------------------------------------------------------------------------#



#Numpy轴
#-----------------------------------------------------------------------------#
"""
# Change False to True for this block of code to see what it does

# NumPy axis argument
if False:
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    print (a.sum())
    print (a.sum(axis=0))
    print (a.sum(axis=1))
    
# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

def min_and_max_riders_per_day(ridership):
    '''
    Fill in this function. First, for each subway station, calculate the
    mean ridership per day. Then, out of all the subway stations, return the
    maximum and minimum of these values. That is, find the maximum
    mean-ridership-per-day and the minimum mean-ridership-per-day for any
    subway station.
    '''
    station_per_day = ridership.mean(axis=0)
    max_daily_ridership = station_per_day.max()  # Replace this with your code
    min_daily_ridership = station_per_day.min()     # Replace this with your code
    
    return (max_daily_ridership, min_daily_ridership)

"""
#-----------------------------------------------------------------------------#


#访问 DataFrame 元素
#-----------------------------------------------------------------------------#

"""
# Subway ridership for 5 stations on 10 different days
ridership_df = pd.DataFrame(
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)

# Change False to True for each block of code to see what it does

# DataFrame creation
if False:
    # You can create a DataFrame out of a dictionary mapping column names to values
    df_1 = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print (df_1)

    # You can also use a list of lists or a 2D NumPy array
    df_2 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=['A', 'B', 'C'])
    print (df_2)
   

# Accessing elements
if False:
    print (ridership_df.iloc[0])
    print (ridership_df.loc['05-05-11'])
    print (ridership_df['R003'])
    print (ridership_df.iloc[1, 3])
    
# Accessing multiple rows
if False:
    print (ridership_df.iloc[1:4])
    
# Accessing multiple columns
if False:
    print (ridership_df[['R003', 'R005']])
    
# Pandas axis
if False:
    df = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print (df.sum())
    print (df.sum(axis=1))
    print (df.values.sum())
    
def mean_riders_for_max_station(ridership):
    '''
    Fill in this function to find the station with the maximum riders on the
    first day, then return the mean riders per day for that station. Also
    return the mean ridership overall for comparsion.
    
    This is the same as a previous exercise, but this time the
    input is a Pandas DataFrame rather than a 2D NumPy array.
    '''
    overall_mean = ridership_df[ridership_df.iloc[0].argmax()].mean() # Replace this with your code
    mean_for_max = ridership_df.values.mean() # Replace this with your code
    
    return (overall_mean, mean_for_max)

"""
#-----------------------------------------------------------------------------#




#皮尔逊积矩相关系数(pearson's r)
#-----------------------------------------------------------------------------#
#1.将各变量标准化
#2. 将每一对数值相乘并计算产品的平均值

#r = average of (x in stdunits) * (y in stdunits) 


#https://classroom.udacity.com/courses/ud170/lessons/5428018709/concepts/54422617800923#
"""
默认情况下，Pandas 的 std() 函数使用贝塞耳校正系数来计算标准偏差。调用 td(ddof=0)
 可以禁止使用贝塞耳校正系数。


filename = 'nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

def correlation(x, y):
    '''
    Fill in this function to compute the correlation between the two
    input variables. Each input is either a NumPy array or a Pandas
    Series.
    
    correlation = average of (x in standard units) times (y in standard units)
    
    Remember to pass the argument "ddof=0" to the Pandas std() function!
    '''
    std_x = (x - x.mean()) / (x.std(ddof=0))
    std_y = (y - y.mean()) / (y.std(ddof=0))
    
    r = (std_x * std_y).mean()
    return r

entries = subway_df['ENTRIESn_hourly']
cum_entries = subway_df['ENTRIESn']
rain = subway_df['meanprecipi']
temp = subway_df['meantempi']

print (correlation(entries, rain))
print (correlation(entries, temp))
print (correlation(rain, temp))

print (correlation(entries, cum_entries))
"""
#-----------------------------------------------------------------------------#





#DataFrame 向量化运算
#-----------------------------------------------------------------------------#
"""
# Examples of vectorized operations on DataFrames:
# Change False to True for each block of code to see what it does

# Adding DataFrames with the column names
if False:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]})
    print (df1 + df2)
    
# Adding DataFrames with overlapping column names 
if False:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df2 = pd.DataFrame({'d': [10, 20, 30], 'c': [40, 50, 60], 'b': [70, 80, 90]})
    print (df1 + df2)

# Adding DataFrames with overlapping row indexes
if False:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]},
                       index=['row1', 'row2', 'row3'])
    df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]},
                       index=['row4', 'row3', 'row2'])
    print (df1 + df2)

# --- Quiz ---
# Cumulative entries and exits for one station for a few hours.
entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})

def get_hourly_entries_and_exits(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits (entries in the first column, exits in the second) and
    return a DataFrame with hourly entries and exits (entries in the
    first column, exits in the second).
    '''
    return entries_and_exits - entries_and_exits.shift(1)

# entries_and_exits.diff 同样可以完成任务
"""
#-----------------------------------------------------------------------------#





#DataFrame applymap()

#-----------------------------------------------------------------------------#


"""
# Change False to True for this block of code to see what it does

# DataFrame applymap()
if False:
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [10, 20, 30],
        'c': [5, 10, 15]
    })
    
    def add_one(x):
        return x + 1
        
    print (df.applymap(add_one))
    
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)
    
    
def convert_grade(grade):
    if grade >= 90:
        return 'A'
    elif grade >= 80:
        return 'B'
    elif grade >= 70:
        return 'C'
    elif grade >= 60:
        return 'D'
    else:   
        return 'F'
def convert_grades(grades):
    '''
    Fill in this function to convert the given DataFrame of numerical
    grades to letter grades. Return a new DataFrame with the converted
    grade.
    
    The conversion rule is:
        90-100 -> A
        80-89  -> B
        70-79  -> C
        60-69  -> D
        0-59   -> F
    '''
    return grades.applymap(convert_grade)


#注意，计算得出的默认标准偏差类型在 numpy 的 .std() 和 pandas 的 .std() 函数之间是
#不同的。默认情况下，numpy 计算的是总体标准偏差，ddof = 0。另一方面，pandas 计算的是
#样本标准偏差，ddof = 1。如果我们知道所有的分数，那么我们就有了总体——因此，要使用 
#pandas 进行归一化处理，我们需要将“ddof”设置为 0。
"""

#-----------------------------------------------------------------------------#





#DataFrame apply() case 1
#-----------------------------------------------------------------------------#


"""
grades_df = pd.DataFrame(
        data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
              'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
        index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio','Fred',
               'Greta', 'Humbert', 'Ivan', 'James'])

# Change False to True for this block of code to see what it does

# DataFrame apply()
if True:
    def convert_grades_curve(exam_grades):
        # Pandas has a bult-in function that will perform this calculation
        # This will give the bottom 0% to 10% of students the grade 'F',
        # 10% to 20% the grade 'D', and so on. You can read more about
        # the qcut() function here:
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
        return pd.qcut(exam_grades,
                       [0, 0.1, 0.2, 0.5, 0.8, 1],
                       labels=['F', 'D', 'C', 'B', 'A'])
        
    # qcut() operates on a list, array, or Series. This is the
    # result of running the function on a single column of the
    # DataFrame.
    print (convert_grades_curve(grades_df['exam1']))
    
    # qcut() does not work on DataFrames, but we can use apply()
    # to call the function on each column separately
    print (grades_df.apply(convert_grades_curve))

def standardize_col(df_col):
    return (df_col-df_col.mean())/df_col.std(ddof = 0)
    
def standardize(df):
    '''
    Fill in this function to standardize each column of the given
    DataFrame. To standardize a variable, convert each value to the
    number of standard deviations it is above or below the mean.
    '''

    return df.apply(standardize_col)


standardize(grades_df)
"""


#-----------------------------------------------------------------------------#




#DataFrame apply() case 2
#-----------------------------------------------------------------------------#
"""
#df.apply(np.max) = df.max()


df = pd.DataFrame({
    'a': [4, 5, 3, 1, 2],
    'b': [20, 10, 40, 50, 30],
    'c': [25, 20, 5, 15, 10]
})

# Change False to True for this block of code to see what it does

# DataFrame apply() - use case 2
if False:   
    print (df.apply(np.mean))
    print (df.apply(np.max))
    
def list_second_largest(dflist):
    df_remove_max = dflist[dflist<dflist.max()]
    df_second = df_remove_max.max()
    return df_second

def second_largest(df):
    '''
    Fill in this function to return the second-largest value of each 
    column of the input DataFrame.
    '''
    return df.apply(list_second_largest)


#另一种解法：

def col_list_largest(col):
    sort_col = col.sort_values(ascending = False)
    return sort_col.iloc[1]

def col_second_largest(df):
    return df.apply(col_list_largest)

"""
#-----------------------------------------------------------------------------#




#向 Series 添加 DataFrame
#-----------------------------------------------------------------------------#
"""
# Change False to True for each block of code to see what it does

# Adding a Series to a square DataFrame
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df + s)
    
# Adding a Series to a one-row DataFrame 
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10], 1: [20], 2: [30], 3: [40]})
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df + s)

# Adding a Series to a one-column DataFrame
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10, 20, 30, 40]})
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df + s)
if False:
    df.add(s,axis = 'index')
    df.add(s)
    
# Adding when DataFrame column names match Series index
if False:
    s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df + s)
    
# Adding when DataFrame column names don't match Series index
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df + s)


"""
#-----------------------------------------------------------------------------#




#再次归一化每一列
#-----------------------------------------------------------------------------#
"""

# Adding using +
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df + s)
    
# Adding with axis='index'
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df.add(s, axis='index'))
    # The functions sub(), mul(), and div() work similarly to add()
    
# Adding with axis='columns'
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print (df)
    print ('') # Create a blank line between outputs
    print (df.add(s, axis='columns'))
    # The functions sub(), mul(), and div() work similarly to add()
    
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def standardize(df):
    '''
    Fill in this function to standardize each column of the given
    DataFrame. To standardize a variable, convert each value to the
    number of standard deviations it is above or below the mean.
    
    This time, try to use vectorized operations instead of apply().
    You should get the same results as you did before.
    '''
    return (df - df.mean())/df.std(ddof = 0)

def standardize_rows(df):
    '''
    Optional: Fill in this function to standardize each row of the given
    DataFrame. Again, try not to use apply().
    
    This one is more challenging than standardizing each column!
    '''
    meandiff = df.sub(df.mean(axis = 1),axis='index')
    return meandiff.div(df.std(axis = 1),axis = 0)


"""
#-----------------------------------------------------------------------------#










#Pandas groupby()
#-----------------------------------------------------------------------------#
"""
#pd.groupby('xxx').groups


values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for each block of code to see what it does

# Examine DataFrame
if False:
    print (example_df)
    
# Examine groups
if False:
    grouped_data = example_df.groupby('even')
    # The groups attribute is a dictionary mapping keys to lists of row indexes
    print (grouped_data.groups)
    
# Group by multiple columns
if False:
    grouped_data = example_df.groupby(['even', 'above_three'])
    print (grouped_data.groups)
    
# Get sum of each group
if False:
    grouped_data = example_df.groupby('even')
    print (grouped_data.sum())
    
# Limit columns in result
if False:
    grouped_data = example_df.groupby('even')
    
    # You can take one or more columns from the result DataFrame
    print (grouped_data.sum()['value'])
    
    print ('\n') # Blank line to separate results
    
    # You can also take a subset of columns from the grouped data before 
    # collapsing to a DataFrame. In this case, the result is the same.
    print (grouped_data['value'].sum())
    
filename = 'nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

### Write code here to group the subway data by a variable of your choice, then
### either print out the mean ridership within each group or create a plot.


#subway_groupdata = subway_df.groupby('day_week').mean()['hour'].mean()
subway_groupdata = subway_df.groupby('day_week').mean()
ridership = subway_groupdata['ENTRIESn_hourly']

ridership.plot()


"""
#-----------------------------------------------------------------------------#



#每小时入站和出站数
#-----------------------------------------------------------------------------#
"""
使用 groupby() 计算每小时入站和出站数
你在之前的测试题中，是对累计入站数的单一集合计算每小时入站和出站数。然而，在原始数据中，
每个站台都有一个单独的数量集。因此，要准确计算每小时入站和出站数，有必要按每天每站台
进行分组，然后计算每天每小时入站和出站数。写下能够完成此操作的一个函数。你应该使用 
apply() 函数来调用你之前写的函数。你还应该确保将分组数据限制在入站和出站两列中，
因为如果你的函数是在非数值型数据类型上被调用，那么它有可能会造成错误。
如果你希望了解在 Pandas 中使用 groupby() 函数的更多信息，可以访问此页面。

注意：你将无法使用此方法，在完整数据集中重新生成 ENTRIESn_hourly 和 EXITSn_hourly 列。
在创建数据集时，我们做了额外的处理，删除了错误值。



values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for each block of code to see what it does

# Standardize each group
if False:
    def standardize(xs):
        return (xs - xs.mean()) / xs.std()
    grouped_data = example_df.groupby('even')
    print (grouped_data['value'].apply(standardize))
    
# Find second largest value in each group
if False:
    def second_largest(xs):
        sorted_xs = xs.sort(inplace=False, ascending=False)
        return sorted_xs.iloc[1]
    grouped_data = example_df.groupby('even')['value']
    print (grouped_data.apply(second_largest))

# --- Quiz ---
# DataFrame with cumulative entries and exits for multiple stations
ridership_df = pd.DataFrame({
    'UNIT': ['R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051'],
    'TIMEn': ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00'],
    'ENTRIESn': [3144312, 8936644, 3144335, 8936658, 3144353, 8936687, 3144424, 8936819, 3144594],
    'EXITSn': [1088151, 13755385,  1088159, 13755393,  1088177, 13755598, 1088231, 13756191,  1088275]
})

def get_hourly_entries_and_exits(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits and return a DataFrame with hourly entries and exits.
    The hourly entries and exits should be calculated separately for
    each station (the 'UNIT' column).
    
    Hint: Use the `get_hourly_entries_and_exits()` function you wrote
    in a previous quiz, DataFrame Vectorized Operations, and the `.apply()`
    function, to help solve this problem.
    '''

    return entries_and_exits.groupby('UNIT')['ENTRIESn','EXITSn'].diff()
"""


#-----------------------------------------------------------------------------#






#合并 Pandas DataFrame
#-----------------------------------------------------------------------------#


"""
subway_df = pd.DataFrame({
    'UNIT': ['R003', 'R003', 'R003', 'R003', 'R003', 'R004', 'R004', 'R004',
             'R004', 'R004'],
    'DATEn': ['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
              '05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ENTRIESn': [ 4388333,  4388348,  4389885,  4391507,  4393043, 14656120,
                 14656174, 14660126, 14664247, 14668301],
    'EXITSn': [ 2911002,  2911036,  2912127,  2913223,  2914284, 14451774,
               14451851, 14454734, 14457780, 14460818],
    'latitude': [ 40.689945,  40.689945,  40.689945,  40.689945,  40.689945,
                  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ],
    'longitude': [-73.872564, -73.872564, -73.872564, -73.872564, -73.872564,
                  -73.867135, -73.867135, -73.867135, -73.867135, -73.867135]
})

weather_df = pd.DataFrame({
    'DATEn': ['05-01-11', '05-01-11', '05-02-11', '05-02-11', '05-03-11',
              '05-03-11', '05-04-11', '05-04-11', '05-05-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'latitude': [ 40.689945,  40.69132 ,  40.689945,  40.69132 ,  40.689945,
                  40.69132 ,  40.689945,  40.69132 ,  40.689945,  40.69132 ],
    'longitude': [-73.872564, -73.867135, -73.872564, -73.867135, -73.872564,
                  -73.867135, -73.872564, -73.867135, -73.872564, -73.867135],
    'pressurei': [ 30.24,  30.24,  30.32,  30.32,  30.14,  30.14,  29.98,  29.98,
                   30.01,  30.01],
    'fog': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'rain': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'tempi': [ 52. ,  52. ,  48.9,  48.9,  54. ,  54. ,  57.2,  57.2,  48.9,  48.9],
    'wspdi': [  8.1,   8.1,   6.9,   6.9,   3.5,   3.5,  15. ,  15. ,  15. ,  15. ]
})

def combine_dfs(subway_df, weather_df):
    '''
    Fill in this function to take 2 DataFrames, one with subway data and one with weather data,
    and return a single dataframe with one row for each date, hour, and location. Only include
    times and locations that have both subway data and weather data available.
    '''
    return pd.merge(subway_df,weather_df,on = ['DATEn','hour','latitude','longitude'],how = 'inner')

"""


#-----------------------------------------------------------------------------#


#使用 DataFrame 绘制图形
#-----------------------------------------------------------------------------#

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for this block of code to see what it does

# groupby() without as_index
if False:
    first_even = example_df.groupby('even').first()
    print (first_even)
    print (first_even['even']) # Causes an error. 'even' is no longer a column in the DataFrame
    
# groupby() with as_index=False
if False:
    first_even = example_df.groupby('even', as_index=False).first()
    print (first_even)
    print (first_even['even']) # Now 'even' is still a column in the DataFrame

filename = 'nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

## Make a plot of your choice here showing something interesting about the subway data.
## Matplotlib documentation here: http://matplotlib.org/api/pyplot_api.html
## Once you've got something you're happy with, share it on the forums!

#x = pd.groupby(subway_df,'DATEn').sum()['day_week']
#x.plot()

data_by_location = subway_df.groupby(['latitude','longitude'],as_index= False).mean()
x= data_by_location['latitude']
y = data_by_location['longitude']
scaled_entries = data_by_location['EXITSn_hourly']/data_by_location['EXITSn_hourly'].std()

plt.scatter(x,y,s = scaled_entries*20)
#-----------------------------------------------------------------------------#

