import pandas as pad
import seaborn as sea
import matplotlib
import plotly.express as px
pad.set_option('display.max_rows',None)
pad.set_option('display.max_columns',None)
pad.set_option('display.width',None)
sea.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 5
matplotlib.rcParams['figure.facecolor'] = '#00000000'

""" Stworzenie data frame'u zawierającego statystyki zawodników z gry Fifa 20"""
df = pad.read_csv (r'../data/players_20.csv')

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



Roma = df[df.club == "Roma"]

R_first = Roma.loc[Roma['player_positions']!='RES']

R_goalk = R_first[R_first['player_positions'].str.contains('GK')]
R_def = R_first[R_first['player_positions'].str.contains('B')]
R_mid = R_first[R_first['player_positions'].str.contains('M')]
searchfor = ['ST', 'LW', 'RW','LM','RM','CF']
R_at = R_first[R_first['player_positions'].str.contains('|'.join(searchfor))]

val_d = pad.DataFrame(R_first)
val_d = R_first['defending'].sum()
mean_defending = val_d/R_first['defending'].size


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

a=overall(Roma)
b=overall(R_at)
c=overall(R_def)
d=overall(R_mid)
e=overall(R_goalk)
f=potential(R_first)

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



print("Skład Romy :")
print(Roma)

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