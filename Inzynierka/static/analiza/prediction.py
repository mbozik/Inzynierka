import pandas as pad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import seaborn as sea
from sklearn.metrics import r2_score
from scipy import stats
from sklearn import metrics

data = pad.read_csv (r'../data/players_20.csv')

data['main_position']=data['player_positions'].str.split(pat=',', n=-1, expand=True)[0]

# data.loc[data['main_position'] == "GK", 'position_category'] = 1
# data.loc[data['main_position'] == "LB", 'position_category'] = 1
# data.loc[data['main_position'] == "LWB", 'position_category'] = 1
# data.loc[data['main_position'] == "RB", 'position_category'] = 1
# data.loc[data['main_position'] == "CB", 'position_category'] = 2
# data.loc[data['main_position'] == "RM", 'position_category'] = 2
# data.loc[data['main_position'] == "RWB", 'position_category'] = 2
# data.loc[data['main_position'] == "LM", 'position_category'] = 2
# data.loc[data['main_position'] == "CDM", 'position_category'] = 3
# data.loc[data['main_position'] == "CM", 'position_category'] = 3
# data.loc[data['main_position'] == "ST", 'position_category'] = 3
# data.loc[data['main_position'] == "CAM", 'position_category'] = 4
# data.loc[data['main_position'] == "LW", 'position_category'] = 4
# data.loc[data['main_position'] == "RW", 'position_category'] = 4
# data.loc[data['main_position'] == "CF", 'position_category'] = 5

data.loc[data['main_position'] == "GK", 'position_category'] = 1
data.loc[data['main_position'] == "LB", 'position_category'] = 1
data.loc[data['main_position'] == "LWB", 'position_category'] = 1
data.loc[data['main_position'] == "RB", 'position_category'] = 1
data.loc[data['main_position'] == "CB", 'position_category'] = 2
data.loc[data['main_position'] == "RM", 'position_category'] = 2
data.loc[data['main_position'] == "RWB", 'position_category'] = 2
data.loc[data['main_position'] == "LM", 'position_category'] = 2
data.loc[data['main_position'] == "CDM", 'position_category'] = 3
data.loc[data['main_position'] == "CM", 'position_category'] = 3
data.loc[data['main_position'] == "CAM", 'position_category'] = 4
data.loc[data['main_position'] == "LW", 'position_category'] = 5
data.loc[data['main_position'] == "RW", 'position_category'] = 5
data.loc[data['main_position'] == "CF", 'position_category'] = 6
data.loc[data['main_position'] == "ST", 'position_category'] = 3

data=data[data.main_position!='GK']

# Skill_cols=['age','position_category','potential', 'international_reputation', 'overall', 'mentality_composure']

Skill_cols=['age', 'height_cm','position_category', 'weight_kg','potential',
       'international_reputation','overall', 'weak_foot', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic',
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'mentality_composure']


# print(np.max([
#     data['player_positions'].apply(lambda x: len(x.split(','))).max(),
#     data['player_positions'].apply(lambda x: len(x.split(','))).max()
# ]))



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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# clf = DecisionTreeClassifier()
# clf.fit(X_train,y_train)
#
# y_pred = clf.predict(X_test)
# mse=mean_squared_error(y_test, y_pred)
# rmse=np.sqrt(mse)
# rmse
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


regresja_liniowa=LinearRegression()
regresja_liniowa.fit(X_train,y_train)

predictions=regresja_liniowa.predict(X_test)
mse=mean_squared_error(y_test, predictions)
rmse=np.sqrt(mse)
rmse
# print("Accuracy:",metrics.accuracy_score(y_test, predictions))

param_grid=[
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8,10,12,14,16,18,30]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4,6]}
]

Las_Losowy_Reg=RandomForestRegressor()

grid_search=GridSearchCV(Las_Losowy_Reg, param_grid, cv=5,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

grid_search.fit(X_train,y_train)


print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))


feature_importances=grid_search.best_estimator_.feature_importances_
features=sorted(zip(feature_importances, Skill_cols),reverse=True)
features_sorted=np.array(features)
features_sorted
print(features_sorted)

plt.pie(features_sorted[:,0], labels=features_sorted[:,1],radius=5,autopct='%1.1f%%')
plt.show()

final_model=grid_search.best_estimator_



def TeamEstimator(team,N=10):
    Players_Team=data[data['club']==team].copy()
    Players_Team_prepared=pipeline.transform(Players_Team)
    Players_prediction=final_model.predict(Players_Team_prepared)
    Players_Team["value_predict"]=Players_prediction
    Players_Team=Players_Team.sort_values(by='value_predict', ascending=False)
    Players_Team["Przewidywana wartość"]=Players_Team["value_predict"].round(2)
    Players_Team["actual_value"]=(Players_Team["value_eur"]/1e6).round(2)
    return (Players_Team[['short_name','wage_eur','actual_value','Przewidywana wartość','age','position_category']].head(N))

def AllEstimator(N=1000):
    All_players=data
    All_players_prepared=pipeline.transform(All_players)
    All_players_prediction=final_model.predict(All_players_prepared)
    All_players["value_predict"]=All_players_prediction
    All_players=All_players.sort_values(by='value_predict', ascending=False)
    All_players["Przewidywana wartość"]=All_players["value_predict"].round(2)
    All_players["actual_value"]=(All_players["value_eur"]/1e6).round(2)
    return (All_players[['short_name','wage_eur','actual_value','Przewidywana wartość','age','position_category']].head(N))



legia = TeamEstimator('Legia Warszawa',N=24)

full = AllEstimator(1000)
sea.lmplot(x="actual_value", y="Przewidywana wartość", data=full)
plt.show()

legia.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
legia.to_csv('../data/legia.csv', index=False,)

mc = TeamEstimator('Manchester City',N=24)
mc.to_csv('../data/mc.csv', index=False,)


roma = TeamEstimator('Roma',N=24)
roma.to_csv('../data/roma.csv', index=False,)

Y_test_prediction=final_model.predict(X_test)
test_mse = mean_squared_error(y_test, Y_test_prediction)
test_rmse = np.sqrt(test_mse)
test_rmse

