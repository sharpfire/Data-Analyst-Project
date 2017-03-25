# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 21:32:43 2017

@author: 呵呵
"""
import pandas as pd
import numpy as np
import seaborn as sns

#x = pd.read_csv(r'C:\Users\yu\Desktop\Learning\Python Learning\udacity-数据分析入门\用NumPy 和 Pandas 分析一维数据\employment_above_15.csv')

#Nummpy数组
#-----------------------------------------------------------------------------#
"""
# First 20 countries with employment data
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])

# Change False to True for each block of code to see what it does

# Accessing elements
if False:
    print (countries[0])
    print (countries[3])

# Slicing
if False:
    print (countries[0:3])
    print (countries[:3])
    print (countries[17:])
    print (countries[:])

# Element types
if False:
    print (countries.dtype)
    print (employment.dtype)
    print (np.array([0, 1, 2, 3]).dtype)
    print (np.array([1.0, 1.5, 2.0, 2.5]).dtype)
    print (np.array([True, False, True]).dtype)
    print (np.array(['AL', 'AK', 'AZ', 'AR', 'CA']).dtype)

# Looping
if False:
    for country in countries:
        print ('Examining country {}'.format(country))

    for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]
        print ('Country {} has employment {}'.format(country,
                country_employment))

# Numpy functions
if False:
    print (employment.mean())
    print (employment.std())
    print (employment.max())
    print (employment.sum())
    
    
def max_employment(countries, employment):
    '''
    Fill in this function to return the name of the country
    with the highest employment in the given employment
    data, and the employment in that country.
    '''
    max_value = 0
    max_country = None 
    for i in range(len(employment)):
        if employment[i] > max_value:
            max_value = employment[i]
            max_country = countries[i]

    return (max_country, max_value)
max_employment(countries, employment)

def numpy_max_employment(countries, employment):
    '''
    Fill in this function to return the name of the country
    with the highest employment in the given employment
    data, and the employment in that country.
    '''
    i = np.argmax(employment)
    max_country = countries[i]
    max_value = employment[i]
    return (max_country, max_value)

numpy_max_employment(countries, employment)

#-----------------------------------------------------------------------------#
"""




#计算整体完成率

#-----------------------------------------------------------------------------#
"""

# First 20 countries with school completion data
countries = np.array([
       'Algeria', 'Argentina', 'Armenia', 'Aruba', 'Austria','Azerbaijan',
       'Bahamas', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Bolivia',
       'Botswana', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi',
       'Cambodia', 'Cameroon', 'Cape Verde'
])

# Female school completion rate in 2007 for those 20 countries
female_completion = np.array([
    97.35583,  104.62379,  103.02998,   95.14321,  103.69019,
    98.49185,  100.88828,   95.43974,   92.11484,   91.54804,
    95.98029,   98.22902,   96.12179,  119.28105,   97.84627,
    29.07386,   38.41644,   90.70509,   51.7478 ,   95.45072
])

# Male school completion rate in 2007 for those 20 countries
male_completion = np.array([
     95.47622,  100.66476,   99.7926 ,   91.48936,  103.22096,
     97.80458,  103.81398,   88.11736,   93.55611,   87.76347,
    102.45714,   98.73953,   92.22388,  115.3892 ,   98.70502,
     37.00692,   45.39401,   91.22084,   62.42028,   90.66958
])

def overall_completion_rate(female_completion, male_completion):
    '''
    Fill in this function to return a NumPy array containing the overall
    school completion rate for each country. The arguments are NumPy
    arrays giving the female and male completion of each country in
    the same order.
    '''
    
    
    a = (female_completion + male_completion)/2
    
    return a

"""
#-----------------------------------------------------------------------------#




#归一化数据
#-----------------------------------------------------------------------------#
"""

# First 20 countries with employment data
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])

# Change this country name to change what country will be printed when you
# click "Test Run". Your function will be called to determine the standardized
# score for this country for each of the given 5 Gapminder variables in 2007.
# The possible country names are available in the Downloadables section.

country_name = 'United States'

def standardize_data(values):
    '''
    Fill in this function to return a standardized version of the given values,
    which will be in a NumPy array. Each value should be translated into the
    number of standard deviations that value is away from the mean of the data.
    (A positive number indicates a value higher than the mean, and a negative
    number indicates a value lower than the mean.)
    '''
    norm = (values - np.mean(values))/np.std(values)
    
    
    return norm
"""


#-----------------------------------------------------------------------------#



#索引数组
#-----------------------------------------------------------------------------#
"""
# Change False to True for each block of code to see what it does

# Using index arrays
if False:
    a = np.array([1, 2, 3, 4])
    b = np.array([True, True, False, False])
    
    print (a[b])
    print (a[np.array([True, False, True, False])])
    
# Creating the index array using vectorized operations
if False:
    a = np.array([1, 2, 3, 2, 1])
    b = (a >= 2)
    
    print (a[b])
    print (a[a >= 2])
    
# Creating the index array using vectorized operations on another array
if False:
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 2, 1])
    
    print (b == 2)
    print (a[b == 2])


def mean_time_for_paid_students(time_spent, days_to_cancel):
    '''
    Fill in this function to calculate the mean time spent in the classroom
    for students who stayed enrolled at least (greater than or equal to) 7 days.
    Unlike in Lesson 1, you can assume that days_to_cancel will contain only
    integers (there are no students who have not canceled yet).
    
    The arguments are NumPy arrays. time_spent contains the amount of time spent
    in the classroom for each student, and days_to_cancel contains the number
    of days until each student cancel. The data is given in the same order
    in both arrays.
    '''
    a = (time_spent[days_to_cancel >= 7]).mean()
    return a

# Time spent in the classroom in the first week for 20 students
time_spent = np.array([
       12.89697233,    0.        ,   64.55043217,    0.        ,
       24.2315615 ,   39.991625  ,    0.        ,    0.        ,
      147.20683783,    0.        ,    0.        ,    0.        ,
       45.18261617,  157.60454283,  133.2434615 ,   52.85000767,
        0.        ,   54.9204785 ,   26.78142417,    0.
])

# Days to cancel for 20 students
days_to_cancel = np.array([
      4,   5,  37,   3,  12,   4,  35,  38,   5,  37,   3,   3,  68,
     38,  98,   2, 249,   2, 127,  35
])

days_to_cancel[days_to_cancel>=7]
"""
#-----------------------------------------------------------------------------#



#+ 和+=运算
#-----------------------------------------------------------------------------#
"""

#code1
a = np.array([1,2,3,4])
b = a 
a += np.array([1,1,1,1])
print (b)
# [2 3 4 5]

#code2

a = np.array([1,2,3,4])
b = a 
a = a + np.array([1,1,1,1])
print (b)
# [1 2 3 4]

#原地与非原地运算
#-----------------------------------------------------------------------------#

#+= 是原地运算，会将所有新值储存在原值得位置，而不是创建新的数组
#+ 不是

a = np.array([1,2,3,4,5])
s = a[:3]
s[0] = 100
print(a)

#array([100,   2,   3,   4,   5])
"""
#-----------------------------------------------------------------------------#





#pd.Series
#-----------------------------------------------------------------------------#
"""
# Accessing elements and slicing
if False:
    print (life_expectancy[0])
    print (gdp[3:6])
    
# Looping
if False:
    for country_life_expectancy in life_expectancy:
        print ('Examining life expectancy {}'.format(country_life_expectancy))
        
# Pandas functions
if False:
    print (life_expectancy.mean())
    print (life_expectancy.std())
    print (gdp.max())
    print (gdp.sum())

# Vectorized operations and index arrays
if False:
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([1, 2, 1, 2])
  
    print (a + b)
    print (a * 2)
    print (a >= 3)
    print (a[a >= 3])
    


countries = ['Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
             'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
             'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia']

life_expectancy_values = [74.7,  75. ,  83.4,  57.6,  74.6,  75.4,  72.3,  81.5,  80.2,
                          70.3,  72.1,  76.4,  68.1,  75.2,  69.8,  79.4,  70.8,  62.7,
                          67.3,  70.6]

gdp_values = [ 1681.61390973,   2155.48523109,  21495.80508273,    562.98768478,
              13495.1274663 ,   9388.68852258,   1424.19056199,  24765.54890176,
              27036.48733192,   1945.63754911,  21721.61840978,  13373.21993972,
                483.97086804,   9783.98417323,   2253.46411147,  25034.66692293,
               3680.91642923,    366.04496652,   1175.92638695,   1132.21387981]

# Life expectancy and gdp data in 2007 for 20 countries
life_expectancy = pd.Series(life_expectancy_values)
gdp = pd.Series(gdp_values)

# Change False to True for each block of code to see what it does

  
def variable_correlation(variable1, variable2):

    v1 = variable1>variable1.mean()
    v2 = variable2>variable2.mean()
    c = v1 == v2
    num_same_direction =c.sum()
    num_different_direction = len(variable1) - num_same_direction
    return (num_same_direction, num_different_direction)

def test_variable_correlation(variable1, variable2):

    both_above = (variable1 > variable1.mean()) & (variable2 > variable2.mean())
    both_below = (variable1 < variable1.mean()) & (variable2 < variable2.mean())
    same_direct = both_above | both_below
    num_same_direction = same_direct.sum()
    num_different_direction = len(variable1) - num_same_direction
    return (num_same_direction, num_different_direction)




test_variable_correlation(gdp,life_expectancy)

"""
#-----------------------------------------------------------------------------#




#Pandas argmax()
#-----------------------------------------------------------------------------#
"""

countries = [
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
]


employment_values = [
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
]

# Employment data in 2007 for 20 countries
employment = pd.Series(employment_values, index=countries)

def max_employment(employment):
    '''
    Fill in this function to return the name of the country
    with the highest employment in the given employment
    data, and the employment in that country.
    
    The input will be a Pandas series where the values
    are employment and the index is country names.
    
    Try using the Pandas argmax() function. Documention is
    here: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.argmax.html
    '''
    max_country =  employment.argmax()   # Replace this with your code
    max_value = employment[employment.argmax()]   # Replace this with your code

    return (max_country, max_value)
"""




#向量运算
#-----------------------------------------------------------------------------#
"""
# Change False to True for each block of code to see what it does

# Addition when indexes are the same
if False:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
    print (s1 + s2)

# Indexes have same elements in a different order
if False:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['b', 'd', 'a', 'c'])
    print (s1 + s2)

# Indexes overlap, but do not have exactly the same elements
if False:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['c', 'd', 'e', 'f'])
    print (s1 + s2)

# Indexes do not overlap
if False:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['e', 'f', 'g', 'h'])
    print (s1 + s2)
"""
#-----------------------------------------------------------------------------#



#向量运算,Pandas Series apply()
#-----------------------------------------------------------------------------#
"""
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([10, 20, 30, 40], index=['c', 'd', 'e', 'f'])

s1.add(s2,fill_value=0)

def add3(x):
    return x +3 

s1.apply(add3())


# Change False to True to see what the following block of code does

# Example pandas apply() usage (although this could have been done
# without apply() using vectorized operations)
if False:
    s = pd.Series([1, 2, 3, 4, 5])
    def add_one(x):
        return x + 1
    print (s.apply(add_one))

names = pd.Series([
    'Andre Agassi',
    'Barry Bonds',
    'Christopher Columbus',
    'Daniel Defoe',
    'Emilio Estevez',
    'Fred Flintstone',
    'Greta Garbo',
    'Humbert Humbert',
    'Ivan Ilych',
    'James Joyce',
    'Keira Knightley',
    'Lois Lane',
    'Mike Myers',
    'Nick Nolte',
    'Ozzy Osbourne',
    'Pablo Picasso',
    'Quirinus Quirrell',
    'Rachael Ray',
    'Susan Sarandon',
    'Tina Turner',
    'Ugueth Urbina',
    'Vince Vaughn',
    'Woodrow Wilson',
    'Yoji Yamada',
    'Zinedine Zidane'
])

def reverse_name(name):
    '''
    Fill in this function to return a new series where each name
    in the input series has been transformed from the format
    "Firstname Lastname" to "Lastname, FirstName".
    
    Try to use the Pandas apply() function rather than a loop.
    '''
    first,last = name.split(' ')
    return last + ',' + first

def reverse_names(names):
    names = names.apply(reverse_name)
    return names
    
"""
#-----------------------------------------------------------------------------#


#在pandas中绘图
#-----------------------------------------------------------------------------#


# The following code reads all the Gapminder data into Pandas DataFrames. You'll
# learn about DataFrames next lesson.

path = r"C:\Users\yu\Desktop\Learning\Python Learning\udacity-数据分析入门\用NumPy 和 Pandas 分析一维数据"
employment = pd.read_csv(path + 'employment_above_15.csv', index_col='Country')
female_completion = pd.read_csv(path + 'female_completion_rate.csv', index_col='Country')
male_completion = pd.read_csv(path + 'male_completion_rate.csv', index_col='Country')
life_expectancy = pd.read_csv(path + 'life_expectancy.csv', index_col='Country')
gdp = pd.read_csv(path + 'gdp_per_capita.csv', index_col='Country')

# The following code creates a Pandas Series for each variable for the United Stataes.
# You can change the string 'United States' to a country of your choice.

employment_us = employment.loc['United States']
female_completion_us = female_completion.loc['United States']
male_completion_us = male_completion.loc['United States']
life_expectancy_us = life_expectancy.loc['United States']
gdp_us = gdp.loc['United States']

# Uncomment the following line of code to see the available country names
# print employment.index.values

# Use the Series defined above to create a plot of each variable over time for
# the country of your choice. You will only be able to display one plot at a time
# with each "Test Run".



