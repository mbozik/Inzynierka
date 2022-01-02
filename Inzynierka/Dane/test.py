from django.http import HttpResponse
from django.shortcuts import render
import json
import plotly

"""import pyodbc as odbc"""
import pandas as pad
import seaborn as sea
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

df = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
data = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')


def club(nazwa):
    return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]


df = club('Paris Saint-Germain')


def rename(data):
    data.rename(columns={'short_name': 'Nazwa zawodnika'}, inplace=True)
    data.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
    data.rename(columns={'value_eur': 'Wartosc'}, inplace=True)
    data.rename(columns={'player_positions': 'Pozycje'}, inplace=True)
    data.rename(columns={'overall': 'Overall'}, inplace=True)
    data.rename(columns={'age': 'Wiek'}, inplace=True)
    return data


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

    return longlist


rename(df)

x = df.nlargest(3, 'Zarobki')
z = x.iloc[0]

z = z['Nazwa zawodnika']

p = cheapReplacement(z)

z = data[data['short_name'] == z][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
# z = z.iloc[0]
rename(z)
z = z.to_html
p = p.to_html

z2 = x.iloc[1]
z2 = z2['Nazwa zawodnika']

p2 = cheapReplacement(z2)
z2 = data[data['short_name'] == z2][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
rename(z2)
# z2 = z2.iloc[1]



z3 = x.iloc[2]
z3 = z3['Nazwa zawodnika']

p3 = cheapReplacement(z3)

z3 = data[data['short_name'] == z3][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]

print(p2)
