import plotly
"""import pyodbc as odbc"""
import pandas as pad
import seaborn as sea
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
pad.set_option('display.max_rows',None)
pad.set_option('display.max_columns',None)
pad.set_option('display.width',None)
import sqlite3
sea.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 5
matplotlib.rcParams['figure.facecolor'] = '#00000000'
from IPython.display import HTML

""" Stworzenie data frame'u zawierającego statystyki zawodników z gry Fifa 20"""
df = pad.read_csv (r'D:\Studia\Praca inżynierska\players_20.csv')



columns = df.columns
abilities = []

# df.hist(bins=40, figsize=(27,17))
# plt.show()

# Przekazanie nazw kolumn do listy abilities
for i in columns:
    abilities.append(i)

print(abilities)
#Wybranie najważniejszych parametrów na podstawie, których zostaną wyłonienie potencjalni zawodnicy

values= df.loc[:,['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]
for i in values.columns:
    df[i].fillna(df[i].mean(),inplace = True)
# Plot dla wzrostu


# sea.pairplot(df[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']])
# plt.show()
# print(df.isnull().sum())

# plt.figure(figsize=(20,8))
# ax = sea.countplot(x='height_cm',data=df)
# plt.show()

#Extract DDG's information, just like we did with the team name before
DDG = df[df['short_name'] == 'De Gea'][['short_name','wage_eur','value_eur','player_positions','overall','age']]

#Assign DDG's wage, position, rating and age to variables
DDGWage = DDG['wage_eur'].item()
DDGPos = DDG['player_positions'].item()
DDGRating = DDG['overall'].item()
DDGAge = DDG['age'].item()

#Wpisanie do MU wszystkich zawodników manchesteru united
MU = df[df.club == "Roma"]
print(MU)


MU_firstEleven = MU.loc[MU['player_positions']!='RES']
#Wypisanie pomocników z głównej jedenastki Manchesteru United
MU_goalkeapers = MU_firstEleven[MU_firstEleven['player_positions'].str.contains('GK')]
print(MU_goalkeapers)
#Wypisanie obrońców z głównej jedenastki Manchesteru United
MU_defenders = MU_firstEleven[MU_firstEleven['player_positions'].str.contains('B')]
print(MU_defenders)

#Wypisanie pomocników z głównej jedenastki Manchesteru United
MU_midfielders = MU_firstEleven[MU_firstEleven['player_positions'].str.contains('M')]
print(MU_midfielders)

#Wypisanie napastników z głównej jedenastki Manchesteru United
searchfor = ['ST', 'LW', 'RW','LM','RM','CF']

MU_attackers = MU_firstEleven[MU_firstEleven['player_positions'].str.contains('|'.join(searchfor))]

print(MU_attackers)

#Obliczenie średniej szybkości zawodników Manchesteru United
val_d = pad.DataFrame(MU_firstEleven)
val_d = MU_firstEleven['defending'].sum()
mean_defending = val_d/MU_firstEleven['defending'].size
# print(mean_defending)

def potential(dane):
    val_p = pad.DataFrame(dane)
    val_p = dane['potential'].sum()
    mean_potential = val_p / dane['potential'].size
    return mean_potential

def overall(dane):
    val_o = pad.DataFrame(dane)
    val_o = dane['overall'].sum()
    mean_overall = val_o / dane['overall'].size
    return mean_overall

a=overall(MU)
b=overall(MU_attackers)
c=overall(MU_defenders)
d=overall(MU_midfielders)
e=overall(MU_goalkeapers)
f=potential(MU_firstEleven)

df = pad.DataFrame(dict(
    r=[a, b, c, d, e, f],
    theta=['Overall druzyny','Overall Atakujacych','Overall Obrońców',
           'Overall Pomocników', 'Overall Bramkarzy', 'Potencjał Drużyny'])
)

fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.update_traces(fill='toself')
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 99],
      color ='black',
      linecolor = 'white',

    )),
  showlegend=False
)

fig.show()

print("Original DataFrame :")
display(MU)

# # Wykres wzrost względem przyśpieszenie
# plt.figure(figsize=(22,8))
# plt.xlabel("Wzrost", fontsize=20)
# plt.xlabel("Przyśpieszenie", fontsize=20)
# plt.title("Porownanie wzrostu wzgledem przyśpieszenia zawodnika", fontsize=20)
# ac=sea.barplot(x='height_cm',y='movement_acceleration', data=df.sort_values('height_cm',inplace=False))
# plt.show()
#
# plt.figure(figsize=(22,8))
# plt.xlabel("Wzrost", fontsize=20)
# plt.xlabel("Szybkość biegu", fontsize=20)
# plt.title("Porownanie wzrostu wzgledem szybkości zawodnika", fontsize=20)
# ac=sea.barplot(x='height_cm',y='movement_sprint_speed', data=df.sort_values('height_cm',inplace=False))
# plt.show()