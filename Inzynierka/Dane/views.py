"""import pyodbc as odbc"""
import pandas as pad

from django.shortcuts import render

# Create your views here.


def home(request):
    return render(
        request,
        'Dane/home.html',
        {
            'title': "Home",
        }
                  )

def clubs(request):


    return render(
        request,
        'Dane/druzyny.html',
        {
            'title': "Clubs",
        }
    )

def rename(data):
        data.rename(columns={'short_name': 'Nazwa zawodnika'}, inplace=True)
        data.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
        data.rename(columns={'value_eur': 'Wartosc'}, inplace=True)
        data.rename(columns={'player_positions': 'Pozycje'}, inplace=True)
        data.rename(columns={'overall': 'Overall'}, inplace=True)
        data.rename(columns={'age': 'Wiek'}, inplace=True)
        return data

def prediction(name):
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
    import statsmodels.api as sm

    data = pad.read_csv('static/data/players_20.csv')
    data.fillna(0)
    data['main_position'] = data['player_positions'].str.split(pat=',', n=-1, expand=True)[0]

    from sklearn.preprocessing import OrdinalEncoder
    ord_enc = OrdinalEncoder()
    data['main_position'] = ord_enc.fit_transform(data[['main_position']])

    Skill_cols = ['age', 'height_cm', 'main_position', 'weight_kg', 'potential',
                  'international_reputation', 'overall', 'wage_eur', 'pace',
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

    pipeline = Pipeline([
        ('transformer', CustomTransformer(Skill_cols)),
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

    X = pipeline.fit_transform(data)
    y = data['value_eur'].copy()
    y = y.values / 1000000

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_sm = sm.add_constant(X_train)
    lr = sm.OLS(y_train, X_train_sm).fit()

    print(lr.summary())

    regresja_liniowa = LinearRegression()
    regresja_liniowa.fit(X_train, y_train)

    predictions = regresja_liniowa.predict(X_test)
    r2 = r2_score(y_test, predictions)

    X2 = pipeline.fit_transform(data)
    y2 = data['value_eur'].copy()
    y2 = y2.values / 1000000

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    regressor = DecisionTreeRegressor(random_state=42)
    regressor = regressor.fit(X2_train, y2_train)

    decision_pred = regressor.predict(X2_test)
    r2_2 = r2_score(y2_test, decision_pred)

    feature_importances = regressor.feature_importances_
    features = sorted(zip(feature_importances, Skill_cols), reverse=True)
    features_sorted = np.array(features)
    features_sorted

    final_model = regressor

    def teamestimator(team):
        players_team = data[data['club'] == team].copy()
        n = len(players_team.index)
        players_team_prepared = pipeline.transform(players_team)
        players_prediction = final_model.predict(players_team_prepared)
        players_team["value_predict"] = players_prediction.round(2)
        players_team.sort_values(by='value_predict', ascending=False)
        players_team["actual_value"] = (players_team["value_eur"] / 1e6).round(2)
        players_team["difference"] = players_team["actual_value"] - players_team["value_predict"]
        return players_team[['short_name', 'wage_eur', 'actual_value', 'value_predict', 'difference', 'age']].head(n)

    club = teamestimator(name)
    club.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
    return(club)


def roma(request):
    df = pad.read_csv('static/data/players_20.csv')
    data = pad.read_csv('static/data/players_20.csv')

    df['player_positions']=df['player_positions'].str.split(pat=',', n=-1, expand=True)[0]
    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Roma')
    df['value_eur'] = (df["value_eur"] / 1e6).round(2).astype(str) + " M Euro"

    def cheapReplacement(player,skillReduction):
        replacee = data[data['short_name'] == player][
            ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]


        replaceePos = replacee['player_positions'].item()
        replaceeWage = replacee['wage_eur'].item()
        replaceeAge = replacee['age'].item()
        replaceeOverall = replacee['overall'].item() - skillReduction


        longlist = data[data['player_positions'] == replaceePos][
            ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]

        removals = longlist[longlist['overall'] <= replaceeOverall].index
        longlist.drop(removals, inplace=True)

        # Repeat for players with higher wages
        removals = longlist[longlist['wage_eur'] > replaceeWage].index
        longlist.drop(removals, inplace=True)

        removals = longlist[longlist['age'] >= replaceeAge].index
        longlist.drop(removals, inplace=True)

        rename(longlist)


        return longlist.sort_values("Zarobki").head(1)

    rename(df)

    x=df.nlargest(3, 'Zarobki')
    zx = x.iloc[0]

    zx=zx['Nazwa zawodnika']


    p=cheapReplacement(zx,2)

    z = data[data['short_name'] == zx][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    # z = z.iloc[0]
    rename(z)
    z=z.to_html
    p=p.to_html

    z2 = x.iloc[1]
    z2 = z2['Nazwa zawodnika']

    p2 = cheapReplacement(z2,0)
    z2 = data[data['short_name'] == z2][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    rename(z2)
    # z2 = z2.iloc[1]
    z2=z2.to_html
    p2=p2.to_html

    z3 = x.iloc[2]
    z3 = z3['Nazwa zawodnika']

    p3 = cheapReplacement(z3,0)
    z3 = data[data['short_name'] == z3][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    rename(z3)
    z3=z3.to_html
    p3=p3.to_html

    roma = prediction('Roma')
    # roma = pad.read_csv('static/data/roma.csv')
    roma.rename(columns={"actual_value": "Aktualna wartość"}, inplace=True)
    roma.rename(columns={'value_predict': 'Przewidywana wartość'}, inplace=True)
    roma.rename(columns={'difference': 'Różnica'}, inplace=True)
    rename(roma)

    roma = roma.sort_values('Zarobki', ascending=False)
    roma = roma.to_html
    html = df.sort_values("Zarobki", ascending = False).to_html
    return render(
        request,
        'Dane/roma.html',
        {
            'title': "Roma",
            'html': html,
            'z': z,
            'p': p,
            'z2': z2,
            'p2': p2,
            'z3': z3,
            'p3': p3,
            'roma': roma,
        }
    )


def manchester(request):
    df = pad.read_csv('static/data/players_20.csv')
    data = pad.read_csv('static/data/players_20.csv')
    df['player_positions'] = df['player_positions'].str.split(pat=',', n=-1, expand=True)[0]
    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Manchester City')
    df['value_eur'] = (df["value_eur"] / 1e6).round(2).astype(str) + " M Euro"

    def cheapReplacement(player, skillReduction):
        replacee = data[data['short_name'] == player][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]


        replaceePos = replacee['player_positions'].item()
        replaceeWage = replacee['wage_eur'].item()
        replaceeAge = replacee['age'].item()
        replaceeOverall = replacee['overall'].item() - skillReduction

        longlist = data[data['player_positions'] == replaceePos][
            ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]

        removals = longlist[longlist['overall'] <= replaceeOverall].index
        longlist.drop(removals, inplace=True)

        # Repeat for players with higher wages
        removals = longlist[longlist['wage_eur'] > replaceeWage].index
        longlist.drop(removals, inplace=True)

        removals = longlist[longlist['age'] >= replaceeAge].index
        longlist.drop(removals, inplace=True)

        rename(longlist)


        return longlist.sort_values("Zarobki").head(1)

    rename(df)

    x=df.nlargest(3, 'Zarobki')
    z = x.iloc[0]

    z=z['Nazwa zawodnika']

    p=cheapReplacement(z,4)

    z = data[data['short_name'] == z][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    # z = z.iloc[0]
    rename(z)
    z=z.to_html
    p=p.to_html

    z2 = x.iloc[1]
    z2 = z2['Nazwa zawodnika']

    p2 = cheapReplacement(z2,2)
    z2 = data[data['short_name'] == z2][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    rename(z2)
    # z2 = z2.iloc[1]
    z2=z2.to_html
    p2=p2.to_html

    z3 = x.iloc[2]
    z3 = z3['Nazwa zawodnika']

    p3 = cheapReplacement(z3,2)

    z3 = data[data['short_name'] == z3][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]

    rename(z3)
    z3=z3.to_html
    p3=p3.to_html

    # mc = pad.read_csv('static\data\mc.csv')
    mc = prediction('Manchester City')
    rename(mc)
    mc = mc.sort_values('Zarobki', ascending=False)
    mc.rename(columns={'value_predict': 'Przewidywana wartość'}, inplace=True)
    mc.rename(columns={'difference': 'Różnica'}, inplace=True)
    mc.rename(columns={"actual_value": "Aktualna wartość"}, inplace=True)
    mc = mc.to_html

    html = df.sort_values("Zarobki", ascending = False).to_html
    return render(
        request,
        'Dane/manchester.html',
        {
            'title': "Manchester City",
            'html': html,
            'z': z,
            'p': p,
            'z2': z2,
            'p2': p2,
            'z3': z3,
            'p3': p3,
            'mc': mc,
        }
    )


def legia(request):
    df = pad.read_csv('static/data/players_20.csv')
    data = pad.read_csv('static/data/players_20.csv')
    df['player_positions'] = df['player_positions'].str.split(pat=',', n=-1, expand=True)[0]
    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Legia Warszawa')

    df['value_eur'] = (df["value_eur"] / 1e6).round(2).astype(str) + " M Euro"

    def cheapReplacement(player, skillReduction=0):
        replacee = data[data['short_name'] == player][
            ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]


        replaceePos = replacee['player_positions'].item()
        replaceeWage = replacee['wage_eur'].item()
        replaceeAge = replacee['age'].item()
        replaceeOverall = replacee['overall'].item() - skillReduction

        longlist = data[data['player_positions'] == replaceePos][
            ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]

        removals = longlist[longlist['overall'] <= replaceeOverall].index
        longlist.drop(removals, inplace=True)

        # Repeat for players with higher wages
        removals = longlist[longlist['wage_eur'] > replaceeWage].index
        longlist.drop(removals, inplace=True)

        removals = longlist[longlist['age'] >= replaceeAge].index
        longlist.drop(removals, inplace=True)
        rename(longlist)

        return longlist.sort_values("Zarobki").head(1)

    rename(df)

    x=df.nlargest(3, 'Zarobki')
    z = x.iloc[0]

    z=z['Nazwa zawodnika']

    #legia  = pad.read_csv('static\data\legia.csv')
    legia = prediction('Legia Warszawa')
    legia = legia.sort_values('Zarobki', ascending=False)
    legia.rename(columns={'value_predict': 'Przewidywana wartość'}, inplace=True)
    legia.rename(columns={'difference': 'Różnica'}, inplace=True)
    legia.rename(columns={"actual_value": "Aktualna wartość"},inplace=True)
    rename(legia)
    legia=legia.to_html

    p=cheapReplacement(z)
    z = data[data['short_name'] == z][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    # z = z.iloc[0]
    rename(z)
    z=z.to_html
    p=p.to_html

    z2 = x.iloc[1]
    z2 = z2['Nazwa zawodnika']

    p2 = cheapReplacement(z2)
    z2 = data[data['short_name'] == z2][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    rename(z2)
    # z2 = z2.iloc[1]
    z2=z2.to_html
    p2=p2.to_html

    z3 = x.iloc[2]
    z3 = z3['Nazwa zawodnika']

    p3 = cheapReplacement(z3)
    z3 = data[data['short_name'] == z3][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    rename(z3)
    z3=z3.to_html
    p3=p3.to_html

    html = df.sort_values("Zarobki", ascending = False).to_html
    return render(
        request,
        'Dane/legia.html',
        {
            'title': "Legia Warszawa",
            'html': html,
            'z': z,
            'p': p,
            'z2': z2,
            'p2': p2,
            'z3': z3,
            'p3': p3,
            'legia': legia,
        }
    )


def general(request):
    return render(
        request,
        'Dane/general.html',
        {
            'title': "General",
            # 'data': data,
        }
    )


def contact(request):

    return render(
        request,
        'Dane/contact.html',
        {
            'title': "Contact",
        }
    )


def club_analisys(request):

    return render(request, 'Dane/clubs.html')

