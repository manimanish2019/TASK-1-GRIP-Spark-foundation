# TASK-1-GRIP-Spark-foundation
Name-Manish kumar Gupta
Data Science and business Analytics Internship
The Sparks Foundation # 'GRIP'
Task1: Prediction of a student marks scored based on number of hours he or she studies.

#Linear Regression with Python Scikit Learn ##
Here in this part we will see how using python Scikit-Learn library for machine learning can be used to 
implement regression functions. 
Since the given task 1 comprises of two variables only so we will go for Simple Linear Regression.
Step 1: Importing the data and all libraries required for this task.
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
15
data1=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data1
print(" Data import is successful")
data1.head(15)
 Data import is successful
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
10	7.7	85
11	5.9	62
12	4.5	41
13	3.3	42
14	1.1	17
#Plotting the data points on 2-D graph to have a look over on the dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

*
# Plotting the distribution of scores
data1.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

Preparing the data
Step 2: Bifurcate the data set into "attributes" (inputs) and "labels" (outputs).
1
X = data1.iloc[:, :-1].values  
y = data1.iloc[:, 1].values
#Now we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:#

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 
Step 3: Algorithm Training
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
​
print("Training is complete.")
Training is complete.
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
​
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

Step 4: Making Prediction
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
y_pred
y_pred
array([16.88414476, 33.73226078, 75.357018  , 26.79480124, 60.49103328])
Step 5: Comparison between actual and predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
a
#  testing  own data
hours = 9.25
own_pred = regressor.predict(np.array(hours).reshape(-1,1))
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
No of Hours = 9.25
Predicted Score = 93.69173248737538
Step 6:Performance Evaluation of the Alogorithm
#The final step is to evaluate the performance of algorithm.

#This step is particularly important to compare how well different algorithms perform on a particular dataset.

from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Squared  Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
Mean Absolute Error: 4.183859899002975
Mean Squared Error: 21.5987693072174
Root Mean Squared  Error: 4.6474476121003665
conclusion
#from the above result we can conclude that student studing for 9.25 hr 's predicted score is 93.69%. hence model did a descent job to predict the student scores.#
