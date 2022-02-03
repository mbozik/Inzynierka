import pandas as pad
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
import statsmodels.api as sm

data = pad.read_csv (r'../data/players_20.csv')
data.fillna(0)
data['main_position']=data['player_positions'].str.split(pat=',', n=-1, expand=True)[0]

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
data['main_position'] = ord_enc.fit_transform(data[['main_position']])

Skill_cols=['age', 'height_cm','main_position', 'weight_kg','potential',
       'international_reputation','overall', 'wage_eur', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic',
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'mentality_composure']

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
    ('transformer', CustomTransformer(Skill_cols)),
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])

X=pipeline.fit_transform(data)
y=data['value_eur'].copy()
y=y.values/1000000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()

print(lr.summary())

regresja_liniowa=LinearRegression()
regresja_liniowa.fit(X_train,y_train)

predictions=regresja_liniowa.predict(X_test)
r2 = r2_score(y_test, predictions)

print('Średni błąd bezwzględny:', metrics.mean_absolute_error(y_test, predictions))
print('Błąd średniokwadratowy:', metrics.mean_squared_error(y_test, predictions))
print('Pierwiastek błędu średniokwadratowego:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Współczynnik determinacji: ',r2)



X2=pipeline.fit_transform(data)
y2=data['value_eur'].copy()
y2=y2.values/1000000

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2,random_state=42)

print('\n')
print('Drzewo decyzyjne')
regressor = DecisionTreeRegressor(random_state=42)
regressor = regressor.fit(X2_train,y2_train)

decision_pred = regressor.predict(X2_test)
r2_2 = r2_score(y2_test,decision_pred)

print('Średni błąd bezwzględny:', metrics.mean_absolute_error(y2_test, decision_pred ))
print('Błąd średniokwadratowy:', metrics.mean_squared_error(y2_test, decision_pred ))
print('Pierwiastek błędu średniokwadratowego:', np.sqrt(metrics.mean_squared_error(y2_test, decision_pred )))
print("Współczynnik determinacji decision tree:",r2_2)

feature_importances=regressor.feature_importances_
features=sorted(zip(feature_importances, Skill_cols),reverse=True)
features_sorted=np.array(features)
features_sorted

final_model=regressor
print(final_model)


def teamestimator(team):
    players_team=data[data['club'] == team].copy()
    n = len(players_team.index)
    players_team_prepared=pipeline.transform(players_team)
    players_prediction=final_model.predict(players_team_prepared)
    players_team["value_predict"]=players_prediction.round(2)
    players_team.sort_values(by='value_predict', ascending=False)
    players_team["actual_value"]=(players_team["value_eur"]/1e6).round(2)
    players_team["difference"] = players_team["actual_value"] - players_team["value_predict"]
    return players_team[['short_name', 'wage_eur', 'actual_value', 'value_predict', 'difference', 'age']].head(n)


def allestimator(n):
    all_players=data
    all_players_prepared=pipeline.transform(all_players)
    all_players_prediction=final_model.predict(all_players_prepared)
    all_players["value_predict"]=all_players_prediction.round(2)
    all_players.sort_values(by='value_predict', ascending=False)
    all_players["actual_value"]=(all_players["value_eur"]/1e6).round(2)
    all_players['difference'] = all_players["actual_value"] - all_players["value_predict"]
    return (all_players[['short_name', 'wage_eur', 'actual_value', 'value_predict', 'difference', 'age']].head(n))

full = allestimator(10000)
# sea.lmplot(x="actual_value", y="Przewidywana wartość", data=full)
# plt.xlabel("Regularna wartość")
#
# plt.show()

legia = teamestimator('Legia Warszawa')
legia.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
legia.to_csv('../data/legia.csv', index=False,)

mc = teamestimator('Manchester City')
mc.to_csv('../data/mc.csv', index=False,)

roma = teamestimator('Roma')
roma.to_csv('../data/roma.csv', index=False,)


