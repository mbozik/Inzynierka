import matplotlib
import pandas as pad
import matplotlib.pyplot as plt
import pandas as pad
import seaborn as sea

pad.set_option('display.max_rows',None)
pad.set_option('display.max_columns',None)
pad.set_option('display.width',None)
sea.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 5
matplotlib.rcParams['figure.facecolor'] = '#00000000'

""" Stworzenie data frame'u zawierającego statystyki zawodników z gry Fifa 19"""
df= pad.read_csv (r'D:\Studia\Praca inżynierska\players_20.csv')

highest_overall_club = df.groupby('club').overall.mean().reset_index().sort_values(by='overall', ascending=False)
highest_overall_club


#Zamiana kilku pozycji zawodnika na jedna wartosc
df['player_positions']=df['player_positions'].str.split(pat=',', n=-1, expand=True)[0]




# top10_clubs = highest_overall_club.head(10)
# plt.figure(figsize = (10,5))
# sea.barplot(x=top10_clubs.overall, y=top10_clubs['club'], palette='dark')
# plt.title("Top 10 najlepszych drużyn w Fifie");


#Usunięcie niepotrzebnych kolumn
df.drop(columns = ['real_face', 'loaned_from'], inplace = True)
df.head()


columns = df.columns
abilities = []

# Przekazanie nazw kolumn do listy abilities
for i in columns:
    abilities.append(i)

# print(abilities)
# print(df['players_positions'].value_counts())

#Sprawdzenie czy są jakieś brakujące Dane
missing_data = pad.isna(df.columns).sum()
missing_data

# #Stworzenie heatmapy prezentującej korelacje pomiędzy statystykami zawodników
# plt.figure(figsize = (25, 25))
# sea.heatmap(df.corr(), annot = True, fmt = '.1f')
# plt.title("Korelacja pomiędzy statystykami zawodników")
# # plt.show()
# #Widać największą korelacje w parametrach bramkarzy .
#
matplotlib.rcParams.update({'font.size': 10})


# plt.figure(figsize= (20, 15))
# ax = sea.countplot(x='Pozycja',data=df,color='red')
# plt.ylabel("Liczba zawodników")
# plt.title("Ilość zawodników na poszczególnych pozycjach")
# plt.show()

# plt.figure(figsize = (16, 8))
# sea.set_style('ticks')
# ax = sea.countplot('player_positions', data = df, palette='crest')
# ax.set_xlabel(xlabel = 'Pozycje zawodników', fontsize = 16)
# ax.set_ylabel(ylabel = 'Liczba zawodników', fontsize = 16)
# ax.set_title(label = 'Ilość zawodników na poszczególnych pozycjach', fontsize = 20)
# # plt.show()



#Wybranie najważniejszych parametrów na podstawie, których zostaną wyłonienie potencjalni zawodnicy

# values= df.loc[:,['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic']]
# for i in values.columns:
#     df[i].fillna(df[i].mean(),inplace = True)
# # Plot dla wzrostu
#
# # print(df.isnull().sum())
#
# plt.figure(figsize=(20,8))
# ax = sea.countplot(x='Height',data=df)
# # plt.show()

# plt.figure(figsize=(22,8))
# sea.countplot(df['preferred_foot'], palette = 'pink')
# ax.set_xlabel(xlabel = 'Lewa', fontsize = 16)
# ax.set_ylabel(ylabel = 'Prawa', fontsize = 16)
# plt.title('Lepsza noga wśród zawodników', fontsize = 20)

# plt.show()


# print(df["player_positions"])
# print(dict.fromkeys(df["player_positions"]))
print(len(df["player_positions"].drop_duplicates()))

# stworzony podział na 4 grupy pozycji



df["value_eur"]=(df["value_eur"]/1e6).round(2)

# Wykres wartości względem pozycji na boisku
plt.figure(figsize=(22,8))
plt.title("Porównanie wartości względem pozycji na boisku", fontsize=20)
ac = sea.barplot(x='player_positions',y='value_eur', data=df.sort_values('value_eur'), color='steelblue')
plt.xlabel("Pozycja zawodnika", fontsize=16)
plt.ylabel("Średnia wartość zawodnika w milionach euro", fontsize=16)
plt.show()


# Wykres wartości względem umiejętności


plt.figure(figsize=(22,8))
plt.title("Porównanie wartości względem umiejętności", fontsize=20)
ac = sea.barplot(x='age',y='value_eur', data=df.sort_values('value_eur'), color='steelblue')
plt.xlabel("Wiek", fontsize=16)
plt.ylabel("Średnia wartość zawodnika w milionach euro", fontsize=16)
plt.show()




# plt.figure(figsize = (18, 8))
# plt.style.use('fivethirtyeight')
# ax = sea.countplot('Position', data = df, palette = 'bone')
# ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
# ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
# ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
# plt.show()


#Diagram przedstawiający kolerację między lepszą nogą a kontrolą nad piłką


plt.rcParams['figure.figsize'] = (16, 8)
sea.lmplot(x = 'skill_ball_control', y = 'dribbling', data = df, col = 'preferred_foot')
# plt.show()