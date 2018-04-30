# Default imports
from greyatomlib.advanced_linear_regression.q01_load_data.build import load_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
np.random.seed(9)
data_set, X_train, X_test, y_train, y_test = load_data('data/house_prices_multivariate.csv')
from greyatomlib.advanced_linear_regression.q02_Max_important_feature.build import Max_important_feature

# Write your solution here
# def polynomial(power=5,random_state=9):
#     Poly_Xtrain = X_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']]
#     Poly_ytrain = y_train
#     poly_features = make_pipeline(PolynomialFeatures(power, include_bias=False), LinearRegression())
#     model = poly_features.fit(Poly_Xtrain, Poly_ytrain)
#     return model

# model = polynomial()
# prediction = model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
#
# print(prediction)



# def polynomial (power= 5,random_state = 9):
#     features = list(Max_important_feature(data_set))
#     #return poly.fit_transform(data_set[features])
#     poly_model = make_pipeline(PolynomialFeatures(power,include_bias=False),
#                            LinearRegression())
#     poly_model.fit( X_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']],y_train)
#     #poly_model.fit(X_train[features],y_train)
#     return poly_model
#
#
# model = polynomial()
# prediction = model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
# print (np.round_(prediction,2), np.array([np.round_(32740.9,2)]))


def polynomial(power=5, random_state=9):
    Poly_Xtrain = X_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']]
    Poly_ytrain = y_train
    poly_features = make_pipeline(PolynomialFeatures(power, include_bias=False), LinearRegression())
    model = poly_features.fit(Poly_Xtrain, Poly_ytrain)
    return model

model = polynomial()
prediction = model.predict(np.array([4, 5, 6, 7]).reshape(1, -1))
print (np.round_(prediction,2), np.array([np.round_(32740.9,0)]))
