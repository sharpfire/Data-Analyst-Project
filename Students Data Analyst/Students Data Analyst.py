# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:48:49 2017

@author: 呵呵
"""


"""
##############################################################################
Table_descriptions
##############################################################################
enrollments.csv：

关于完成第一个项目的Data Analyst Nanodegree学生的随机子集的数据，以及没完成第一个
项目的学生的随机子集。

列：
    - account_key：注册学生帐户的唯一标识符。

     - status：收集数据时学生的注册状态。 可能的值为“cancelled”和“current”。

     - join_date：学生注册的日期。

     - cancel_date：学生取消的日期，如果学生尚未取消，则为空。

     - days_to_cancel：join_date和cancel_date之间的天数，如果学生尚未取消，则为空。

     - is_udacity：如果帐户是Udacity测试帐户，则为True，否则为False。

     - is_canceled：如果学生在收集数据时取消了此注册，则为True，否则为False。

-------------------------------------------------------------------------------
daily_engagement.csv:

关于每个学生在注册时注册表中的数据分析师Nanodegree课程中参与的数据。包括记录，即使当
天没有参与。包括Nanodegree计划支持课程的互动数据，以及相同内容的相应免费提供的课程。

列：
     - acct：学生的帐户的唯一标识符，其参与数据是。

     - utc_date：收集数据的日期。

     - num_courses_visited：学生在此日的2分钟内访问的Data Analyst Nanodegree课程
                            总数.Nanodegree课程和具有相同内容的免费课程将单独计算。

     - total_minutes_visited：学生在这一天花费Data Analyst Nanodegree课程的总分钟数。

     - lessons_completed：当天Data Analyst Nanodegree课程中的总课程数。

     - projects_completed：学生在此日完成的Data Analyst Nanodegree项目的总数。

-------------------------------------------------------------------------------
project_submissions.csv:

关于每个学生的Data Analyst Nanodegree项目提交的数据注册表。

列：
     - creation_date：项目提交的日期。

     - completion_date：评估项目的日期。

     - assigned_rating：此列具有4个可能的值：
                        空白 - 项目尚未评估。
                        INCOMPLETE - 项目不符合规格。
                        PASSED - 项目符合规范。
                        DISTINCTION - 项目超出规格。
                        UNGRADED - 无法评估提交的内容
                                   （例如包含损坏的文件）

     - account_key：学生帐户的唯一标识符提交项目。

     - lesson_key：提交的项目的唯一标识符。

     - processing_state：此列具有2个可能的值：
                        CREATED - 项目已提交但未评估。
                        EVALUATED  - 项目已经过评估。
-------------------------------------------------------------------------------

daily_engagement_full.csv:

Similar to daily_engagement.csv, but with engagement further broken down by
course and with more columns available. This file is about 500 megabytes, which
is why the smaller daily_engagement.csv file was created. This dataset is
optional; it is not needed to complete the course.

In addition to the following columns, this table also contains all the same
columns as daily_engagement.csv, except with has_visited instead of
num_courses_visited.

Columns:
    - registration_date:  Date the account was registered.

    - subscription_start: Date paid subscription for the account started.

    - course_key:         Course in which activity is recorded.

    - sibling_key:        Free course with the same free content as course_key.
                          If course_key is a free course, course_key and
                          sibling_key are the same.

    - course_title:       Title of the course.

    - has_visited:        1 if the student visited this course for at least 2
                          minutes on this day.
##############################################################################
"""

import pandas as pd
import unicodecsv
from datetime import datetime as dt

#a = r'C:\Users\yu\Desktop\Learning\Python Learning\udacity-数据分析入门\daily_engagement.csv'
#file = pd.read_csv(a)

#LOAD DATA
#------------------------------------------------------------------------------#
def getcsv(fileplace):
    f = open(fileplace,'rb')
    reader = unicodecsv.DictReader(f)
    return list(reader)
    
enrollments = getcsv(r'C:\Users\yu\Desktop\Learning\Python Learning\udacity-数据分析入门\enrollments.csv')
daily_engagement = getcsv(r'C:\Users\yu\Desktop\Learning\Python Learning\udacity-数据分析入门\daily_engagement.csv')
project_submissions = getcsv(r'C:\Users\yu\Desktop\Learning\Python Learning\udacity-数据分析入门\project_submissions.csv')

#------------------------------------------------------------------------------#


#Fixing Data Types
#------------------------------------------------------------------------------#
def parse_date(date):
    if date == '' :
        return None 
    else:
        return dt.strptime(date,'%Y-%m-%d')

def paser_int(number):
    if number == '' :
        return None
    else:
        return int(float(number))
def fixdatatype():
    
    for enrollment in enrollments:
        enrollment['account_key'] = paser_int(enrollment['account_key'])
        enrollment['days_to_cancel'] = paser_int(enrollment['days_to_cancel'])
        enrollment['cancel_date'] = parse_date(enrollment['cancel_date'])
        enrollment['join_date'] = parse_date(enrollment['join_date'])
        enrollment['is_canceled'] = enrollment['is_canceled'] == 'True'
        enrollment['is_udacity'] = enrollment['is_udacity'] == 'True'
        
    for daily_engage in daily_engagement :
        daily_engage['acct'] = paser_int(daily_engage['acct'])
        daily_engage['lessons_completed'] = paser_int(daily_engage['lessons_completed'])
        daily_engage['num_courses_visited'] = paser_int(daily_engage['num_courses_visited'])
        daily_engage['projects_completed'] = paser_int(daily_engage['projects_completed'])
        daily_engage['total_minutes_visited'] = float(daily_engage['total_minutes_visited'])
        daily_engage['utc_date'] = parse_date(daily_engage['utc_date'])
        
    for project_submission in project_submissions :
        project_submission['account_key'] = paser_int(project_submission['account_key'])
        project_submission['completion_date'] = parse_date(project_submission['completion_date'])
        project_submission['creation_date'] = parse_date(project_submission['creation_date'])
        project_submission['lesson_key'] = paser_int(project_submission['lesson_key'])

fixdatatype()

#------------------------------------------------------------------------------#




#Data explore & Ask The right Question
#------------------------------------------------------------------------------#
#好奇心
#Question 1: 学生需要多长时间提交项目
#Question 2: 找出通过和未通过项目的学生 有什么不同
#Question 3: 退出的学生一般使用了多少天
#Question 4: 留存的学生一般使用了多少天
#Question 5: 退出学生和留存学生访问的课程数量
#Question 6: 学生平均上课时间
#Question 7: 上课时间，上课数量与完成项目间的关系
#Question 8: 参与度随时间变化的关系

#关注问题：
#Question 1: 通过和未通过首个项目的学生每天的参与次数有何不同

def find_unique(data,account = 'account_key'):
    unique_students = set()
    for i in data:
        unique_students.add(i[account])
    return list(unique_students)


def print_unique():
    print ('########','\ndaily_engagement')
    print (len(daily_engagement))
    print (len(find_unique(daily_engagement,'acct')))

    print ('########','\nenrollments')
    print (len(enrollments))
    print (len(find_unique(enrollments,'account_key')))
   
    print ('########','\nproject_submissions')  
    print (len(project_submissions))
    print (len(find_unique(project_submissions,'account_key')))

print_unique()


#发现问题：
#Question 1: 在enrollments中的学生数量比daily_engagement中多？
#Question 2: account_key 和 acct 容易弄混 


#解决Question 2:
#------------------------------------------------------------------------------#
def rename(data,before_name,after_name):
    for i in range(len(data)):
        data[i][after_name] = data[i][before_name]
        del data[i][before_name]


rename(daily_engagement,'acct','account_key')
#------------------------------------------------------------------------------#



#解决Question 1:
#------------------------------------------------------------------------------#
#1.找出没有相应参与数据的注册记录
#2.打印一个或多个数据异常点，有时可以直接观察

unique_enrollments_students  = find_unique(enrollments,'account_key')
unique_daily_engagement_students  = find_unique(daily_engagement,'account_key')


"""
stulist = findunusual(unique_enrollments_students,unique_daily_engagement_students)

for i in stulist:
    for k in enrollments:
        if enrollments[k]['account_key'] == i :
            print (enrollments[k])
"""

def findstu():
    x = []
    for enrollment in enrollments:
        student = enrollment['account_key']
        if student not in unique_daily_engagement_students:
            x.append(enrollment)
    return x

stu = findstu()

#结论：daily_engagement中缺少的部分是因为 enrollment中的学生当天注册当天就注销了
#     不是什么问题


#解决完Q1Q2后，继续探索
#------------------------------------------------------------------------------#
#继续检查注册表中注册至少一天的学生未出现在参与表中的数量。
def findstu_MT1():
    x = []
    for i in range(len(stu)):
        
        if stu[i]['days_to_cancel'] != 0 :
            x.append(stu[i])
    return x

morestu = findstu_MT1()

#结论 :  最终结果有3个账户未出现，morestu 中is_udacity代表是测试账户，测试账户不一
#        定会在daily_engagement中出现，然后删除所有测试账户

#删除daily_engagement，enrollments，project_submissions中测试账户的数据
#------------------------------------------------------------------------------#


def find_test():
    x = set()
    for enrollment in enrollments:
        if enrollment['is_udacity']:
            x.add(enrollment['account_key'])
    return list(x)

test_acct = find_test()



def DEL_TEST(data,test_list):
    x = []
    for i in data:
        if not i['account_key'] in test_list:
            x.append(i)
    return x 
            
enrollments = DEL_TEST(enrollments,test_acct)
daily_engagement = DEL_TEST(daily_engagement,test_acct)
project_submissions = DEL_TEST(project_submissions,test_acct)
#------------------------------------------------------------------------------#




#完成数据再加工 继续探索数据

#Question ：通过第一个项目的学生在每日参与表中的表现有何不同
#问题不够明确
#1：daily中会包含提交项目后的数据，只需观察首个项目之前的表现
#2：如果只看第一次提交前的参与情况，那么比较的是不同时间段的参与
#   数据，参与度可能会发生变化，1个学生1个月的表现与另1个学生2个
#   月的表现相比，可能会有误导性
#3：我们使用的daily表中包含整个项目的参与数据，也包含了与项目无关
#   的参与数据

#解决问题：解决1,2问题，只需要查看学生注册第一周的数据，把一周内
#         注销的学生排除，时间段相同，还可以排除7天免费时间的学生
#         解决3问题，查看


#解决方案：
#1、创建未注销学生的字典 或注销前注册长达7天的学生 (days_to_cancel == None days_to_cancel >=7)) 

#key: acct ;value : join_date
def paid_students():
    
    paid_students = {}
    for enrollment in enrollments:
        if (not enrollment['is_canceled'] or
                enrollment['days_to_cancel'] > 7):
            account_key = enrollment['account_key']
            enrollment_date = enrollment['join_date']
            if (account_key not in paid_students or
                    enrollment_date > paid_students[account_key]):
                    paid_students[account_key] = enrollment_date
    return paid_students

paid_students = paid_students()



"""
5.10. Boolean operations
or_test  ::=  and_test | or_test "or" and_test
and_test ::=  not_test | and_test "and" not_test
not_test ::=  comparison | "not" not_test


In the context of Boolean operations, and also when expressions are used by 
control flow statements, the following values are interpreted as false: False,
 None, numeric zero of all types, and empty strings and containers (including 
 strings, tuples, lists, dictionaries, sets and frozensets). All other values
 are interpreted as true. (See the __nonzero__() special method for a way to 
 change this.)

The operator not yields True if its argument is false, False otherwise.

The expression x and y first evaluates x; if x is false, its value is returned;
 otherwise, y is evaluated and the resulting value is returned.

The expression x or y first evaluates x; if x is true, its value is returned; 
otherwise, y is evaluated and the resulting value is returned.

(Note that neither and nor or restrict the value and type they return to False 
and True, but rather return the last evaluated argument. This is sometimes 
useful, e.g., if s is a string that should be replaced by a default value if 
it is empty, the expression s or 'foo' yields the desired value. Because not 
has to invent a value anyway, it does not bother to return a value of the same
 type as its argument, so e.g., not 'foo' yields False, not ''.)
"""

def within_one_week(join_date, engagement_date):
    time_delta = engagement_date - join_date
    return time_delta.days >= 0 and time_delta.days < 7

def remove_free_trial_cancels(data):
    new_data = []
    for data_point in data:
        if data_point['account_key'] in paid_students:
            new_data.append(data_point)
    return new_data

enrollments = remove_free_trial_cancels(enrollments)
daily_engagement = remove_free_trial_cancels(daily_engagement)
project_submissions = remove_free_trial_cancels(project_submissions)

for engagement in daily_engagement:
    if engagement['num_courses_visited'] > 0 :
        engagement['has_visited'] = 1
    else:
        engagement['has_visited'] = 0

    


print (len(enrollments))
print (len(daily_engagement))
print (len(project_submissions))


def paid_engagement_in_fk():
    
    paid_engagement_in_first_week = []
    for engagement_record in daily_engagement:
        account_key = engagement_record['account_key']
        join_date = paid_students[account_key]
        engagement_record_date = engagement_record['utc_date']
    
        if within_one_week(join_date, engagement_record_date):
            paid_engagement_in_first_week.append(engagement_record)
    return paid_engagement_in_first_week

paid_engagement_in_first_week = paid_engagement_in_fk()

#接下来将数据分为两组，一组是完成项目的，一组是未完成的。

#学生第一周上课的时间



from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
# Create a dictionary of engagement grouped by student.
# The keys are account keys, and the values are lists of engagement records.

def data_group(data,key):
    
    grouped_data = defaultdict(list)
    for datapoints in data:
        account_key = datapoints[key]
        grouped_data[account_key].append(datapoints)
    return grouped_data

def sum_field(grouped_data,field_name):
    
    field_by_account = {}
    for account_key, data in grouped_data.items():
        total_field = 0
        for field in data:
            total_field += field[field_name]
        field_by_account[account_key] = total_field
                        
    return field_by_account

def describe_data(data):
    print ('Mean:', np.mean(data))
    print ('Standard deviation:', np.std(data))
    print ('Minimum:', np.min(data))
    print ('Maximum:', np.max(data))
    plt.hist(data)
# Summarize the data about minutes spent in the classroom

engagement_by_account = data_group(paid_engagement_in_first_week,'account_key')





total_minutes = sum_field(engagement_by_account,'total_minutes_visited')
total_lessons = sum_field(engagement_by_account,'lessons_completed')

describe_data(list(total_minutes.values()))
describe_data(list(total_lessons.values()))



#发现问题，Maximum超出了一周的总分钟数，并且标准差远超均值
#重新修正函数，解决问题：within_one_week 添加条件，time_delta.days >= 0
"""
max(total_minutes_by_account.items(), key=lambda pair: pair[1])

test code 
a =  ([x for x in daily_engagement if x['account_key'] == 108 ])
b =  ([x for x in paid_engagement_in_first_week if x['account_key'] == 108 ])
"""

#分析各学生上课的总天数，不考虑上课的数量,在sum_field函数中添加

total_days = sum_field(engagement_by_account,'has_visited')
describe_data(list(total_days.values()))




## Create two lists of engagement data for paid students in the first week.
## The first list should contain data for students who eventually pass the
## subway project, and the second list should contain data for students
## who do not.


#按照课程号及rating数据，将project_submissions数据分为通过项目的学生和未通过项目的学生
#------------------------------------------------------------------------------#
subway_project_lesson_keys = [746169184, 3176718735]


def subway_project():
    subway_project = []
    pass_students = set()
    for passed_stu in project_submissions:
        if (passed_stu['lesson_key'] in subway_project_lesson_keys) and (passed_stu['assigned_rating'] == 'PASSED' or passed_stu['assigned_rating'] == 'DISTINCTION') :
            subway_project.append(passed_stu)
            pass_students.add(passed_stu['account_key'])
    return subway_project,pass_students

subway_project = subway_project()
pass_list = list(subway_project[1])
"""
def split_pass_stu(data,passlist):
    pass_stu = []
    nonepass_stu = []
    for key,val in engagement_by_account.items():
        if key in pass_list:
            pass_stu.append(val)
        else:
            nonepass_stu.append(val)
            
    return pass_stu,nonepass_stu
"""      
def split_pass_stu(data,passlist):
    pass_stu = []
    nonepass_stu = []
    for i in data:
        if i['account_key'] in pass_list:
            pass_stu.append(i)
        else:
            nonepass_stu.append(i)
            
    return pass_stu,nonepass_stu


pass_stu,nonepass_stu = split_pass_stu(paid_engagement_in_first_week,pass_list)
#------------------------------------------------------------------------------#

#完成已通过项目和未通过项目的学生的参与情况
#------------------------------------------------------------------------------#
print(len(pass_stu),len(nonepass_stu))

pass_stu = data_group(pass_stu,'account_key')
nonepass_stu = data_group(nonepass_stu,'account_key')

pass_stu_min = sum_field(pass_stu,'total_minutes_visited')
nonepass_stu_min = sum_field(nonepass_stu,'total_minutes_visited')

pass_stu_lessons = sum_field(pass_stu,'lessons_completed')
nonepass_stu_lessons = sum_field(nonepass_stu,'lessons_completed')

pass_stu_visited = sum_field(pass_stu,'has_visited')
nonepass_stu_visited = sum_field(nonepass_stu,'has_visited')


describe_data((list(pass_stu_min.values())))
describe_data((list(nonepass_stu_min.values())))


describe_data((list(pass_stu_lessons.values())))
describe_data((list(nonepass_stu_lessons.values())))

describe_data((list(pass_stu_visited.values())))
describe_data((list(nonepass_stu_visited.values())))
#------------------------------------------------------------------------------#


#数据可视化
#------------------------------------------------------------------------------#





#数据t探索结论：
#------------------------------------------------------------------------------#
#未经检验的初步结论： 通过地铁项目的学生在第一周上课的分钟数比没通过的多
#（尽量用统计学去解决问题，也可能是数据噪音造成的）

#另一类初步结论： 如果改变某一数据，另一数据也会变化，例如通过首个项目的学生更愿意在
#                第一周多上课（关联关系）

#另一类初步结论： 通过发邮件提示学生回来上课是否能够提高学生通过项目的数量？（因果关系）

#很多情况下，关联关系并不意味着有因果关系，也可能是第三类因素导致会导致 
#上课数量 和 通过的项目数量共同上升
 
#比如说： 1，对课程的兴趣 2，背景知识，没人有数据分析的背景指数 
#所以需要进行试验 来验证两者是否是因果关系，A/B test课程
#------------------------------------------------------------------------------#




#做出预测
#------------------------------------------------------------------------------#
# 用机器学习去进行预测，#机器学习介绍课程

#------------------------------------------------------------------------------#





#分享研究成果
#------------------------------------------------------------------------------#

#找出你觉得最有趣的地方：
#different in total minutes
#different in days visited 


#找出最好展现的地方，可视化：

#report aveerage 
#show histograms
#------------------------------------------------------------------------------#



#分享研究成果
#------------------------------------------------------------------------------#



import seaborn as sns

plt.hist(list(pass_stu_visited.values()), bins=8)
plt.xlabel('Number of days')
plt.title('Distribution of classroom visits in the first week ' + 
          'for students who do not pass the subway project')

plt.hist(list(nonepass_stu_visited.values()), bins=8)
plt.xlabel('Number of days')
plt.title('Distribution of classroom visits in the first week ' + 
          'for students who pass the subway project')




#数据分析，术语
#------------------------------------------------------------------------------#

#Data science 
#数据科学与数据分析很相似，但有区别
#数据科学与构建推荐系统，排序算法等

#Data Engineering
#主要关注数据再加工，数据分析程序中数据再加工的阶段，负责制作数据管道
#更多的涉及数据的存储和处理

#Big Data
# 忽悠

















