"""

StreetEasy is New York City's leading real estate marketplace.

A Multiple Linear Regression will be applied to rental listings in Manhattan
Brooklyn, and Queens. The code below finds the correlations between several features and the rent, build/evalute
a MLR model, and use the model to present interesting findings:

Does having a washer/dryer in unit increase the price of rent?

How costly is living by a subway station in Brooklyn/Queens?

Is a renant over or underpaying?

Data: www.streeteasy.com/rentals

"""


import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Pulls StreetEasy's Brooklyn renntal listings
streeteasy = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/brooklyn.csv")

#Turns it into a 2 dimentional DataFrame
df = pd.DataFrame(streeteasy)

#Assings the independent variables and features that I think have influence on rent
x = df[['bedrooms', 'bathrooms', 'size_sqft',
       'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck',
       'has_washer_dryer', 'has_elevator', 'has_dishwasher',
       'has_patio']]

#My dependent variable AKA what i'm interested about
y = df[['rent']]

#Use scikit-learn's train_test_split method to split x into 80% training set
#and 20% testing set
x_train, x_test, y_train, y_test = train_test_split(x ,y ,train_size= 0.8, test_size = 0.2, random_state=6)

print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

#create the create
mlr = LinearRegression()

#Fit the model
mlr.fit(x_train, y_train)

#use the model to predit y-values from x_test. Store the predictions in a variable called y_predict
y_predict = mlr.predict(x_test)


#Let's see this model in action, I'm testing it on my friend's Sonny's apartment's as an example
#To see if he's over or underpaying!!
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1]]

predict = mlr.predict(sonny_apartment)

print("Prediction rent: $%.2f" % predict )
plt.scatter(y_test, y_predict, alpha=0.4)

plt.xlabel("Prices: StreetEasy Price")
plt.ylabel("Predicted prices: Actual worth based on model")
plt.title("Actual Rent vs Predicted Rent")

plt.show()

"""
Let's gain more insight from the MLG and our independent variables by creating
a 2D scatterplot to see how the independent variables impact prices.


"""


"""
With this basic Multiple Linear Regression model I'll tune and evaluate it.
This will be done using sklearn's .fit() method giving me the coefficients and
intercept.

"""
print(mlr.coef_)

"""
Graphs can also be generated to evaluate whether any of the independent variables
have a strong positive correlation with rent.

"""
plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)

plt.xlabel("Size Sqft")
plt.ylabel("Rent")
plt.title("Linear Relationship between Size Sqft and Rent")

plt.show()

"""
Time to evaluate the accuracy of my model multiple linear regression model using
Residual Analysis.

"""
mlr_train_score = mlr.score(x_train, y_train)

print("Train score:" + str(mlr_train_score))

mlr_test_score = mlr.score(x_test, y_test)

print("Test score:" + str(mlr_test_score))


"""

Uncomment the code below and enter your own apartment details based off the
variables I chose above

"""

#your_aparment = [[ , , , , , , , , , , , ]]
#predict = model.prediction(you_aparment)
#print("Prediction rent: $%.2f" % predict )
