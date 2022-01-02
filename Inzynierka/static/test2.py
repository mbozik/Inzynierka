import pandas as pad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import patsy
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pad
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pad.read_csv (r'players_20.csv')

data['main_position']=data['player_positions'].str.split(pat=',', n=-1, expand=True)[0]

# data.rename(columns={'main_position': 'Pozycja zawodnika'}, inplace=True)
# data.rename(columns={'value_eur': 'Wartosc'}, inplace=True)
# data.rename(columns={'age': 'Wiek'}, inplace=True)
#
#
# Players=data.groupby('Pozycja zawodnika')['Wartosc'].mean()/1e6
# Players=Players.sort_values()
# Players.plot(kind="bar",figsize=(12,8),color='grey')
# plt.xlabel("Przeciętna wartość zawodnika w milionach")
# plt.show()
#
#
# Players_age=data.groupby('Wiek')['Wartosc'].mean()/1e6
# Players_age.plot(grid=True,figsize=(12,8),color='grey')
# plt.ylabel('Przeciętna wartość zawodnika w milinoach')
# plt.xlabel('Wiek')
# plt.show()

data=data[data.main_position!='GK']
Skill_cols=['age', 'height_cm', 'weight_kg','potential',
       'international_reputation', 'weak_foot', 'skill_moves', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic',
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'mentality_composure', 'defending_marking', 'defending_standing_tackle',
       'defending_sliding_tackle']
print(len(Skill_cols))

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, Columns):
        self.Columns = Columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        New_X = X.copy()
        New_X = New_X[self.Columns].copy()
        return New_X



pipeline=Pipeline([
    ('custom_tr', CustomTransformer(Skill_cols)),
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])

X=pipeline.fit_transform(data)
y=data['value_eur'].copy()
y=y.values/1000000



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)



predictions=lin_reg.predict(X_test)
mse=mean_squared_error(y_test, predictions)
rmse=np.sqrt(mse)
rmse


param_grid=[
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8,10]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4,6]}
]
forest_reg=RandomForestRegressor()

grid_search=GridSearchCV(forest_reg, param_grid, cv=5,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

grid_search.fit(X_train,y_train)

print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))


feature_importances=grid_search.best_estimator_.feature_importances_
features=sorted(zip(feature_importances, Skill_cols),reverse=True)
features_sorted=np.array(features)
features_sorted


plt.pie(features_sorted[:,0], labels=features_sorted[:,1],radius=5,autopct='%1.1f%%')
# plt.show()

final_model=grid_search.best_estimator_

def NationalTeamEstimator(nation,N=10):
    Players_National=data[data['club']==nation].copy()
    Players_National_prepared=pipeline.transform(Players_National)
    National_prediction=final_model.predict(Players_National_prepared)
    Players_National["value_predict"]=National_prediction
    Players_National=Players_National.sort_values(by='value_predict', ascending=False)
    Players_National["Model prediction"]=Players_National["value_predict"].round(2).astype(str)+" M Euro"
    Players_National["actual_value"]=(Players_National["value_eur"]/1e6).round(2).astype(str)+" M Euro"
    return (Players_National[['long_name','nationality','age','club','actual_value','Model prediction']].head(N))


print(NationalTeamEstimator('Legia Warszawa',N=20))



Y_test_prediction=final_model.predict(X_test)
test_mse = mean_squared_error(y_test, Y_test_prediction)
test_rmse = np.sqrt(test_mse)
test_rmse

