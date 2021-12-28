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
df= pad.read_csv (r'D:\Studia\Praca inżynierska\players_19.csv')


#Usunięcie niepotrzebnych kolumn
df.drop(columns = ['Real Face', 'Loaned From'], inplace = True)
df.head()


columns = df.columns
abilities = []

# Przekazanie nazw kolumn do listy abilities
for i in columns:
    abilities.append(i)

print(abilities)
print(df['Position'].value_counts())
#Sprawdzenie czy są jakieś brakujące Dane
missing_data = pad.isna(df.columns).sum()
missing_data

#Stworzenie heatmapy prezentującej korelacje pomiędzy statystykami zawodników
plt.figure(figsize = (25, 25))
sea.heatmap(df.corr(), annot = True, fmt = '.1f')
plt.title("Korelacja pomiędzy statystykami zawodników")
# plt.show()
#Widać największą korelacje w parametrach bramkarzy .

matplotlib.rcParams.update({'font.size': 10})


# plt.figure(figsize= (20, 15))
# ax = sea.countplot(x='Pozycja',data=df,color='red')
# plt.ylabel("Liczba zawodników")
# plt.title("Ilość zawodników na poszczególnych pozycjach")
# plt.show()

plt.figure(figsize = (16, 8))
sea.set_style('ticks')
ax = sea.countplot('Position', data = df, palette='crest')

ax.set_xlabel(xlabel = 'Pozycje zawodników', fontsize = 16)
ax.set_ylabel(ylabel = 'Liczba zawodników', fontsize = 16)
ax.set_title(label = 'Ilość zawodników na poszczególnych pozycjach', fontsize = 20)
plt.show()

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

plt.rcParams['figure.figsize'] = (10, 5)
sea.countplot(df['Preferred Foot'], palette = 'pink')
plt.title('Lepsza noga wśród zawodników', fontsize = 20)
plt.show()

#Diagram przedstawiający kolerację między lepszą nogą a kontrolą nad piłką


plt.rcParams['figure.figsize'] = (16, 8)
sea.lmplot(x = 'BallControl', y = 'Dribbling', data = df, col = 'Preferred Foot')

plt.show()