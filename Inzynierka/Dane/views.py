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
        'Dane/clubs.html',
        {
            'title': "Clubs",
        }
    )


def roma(request):
    df = pad.read_csv('static/data/players_20.csv')
    data = pad.read_csv('static/data/players_20.csv')
    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Roma')
    df['value_eur'] = (df["value_eur"] / 1e6).round(2).astype(str) + " M Euro"

    def rename(data):
        data.rename(columns={'short_name': 'Nazwa zawodnika'}, inplace=True)
        data.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
        data.rename(columns={'value_eur': 'Wartosc'}, inplace=True)
        data.rename(columns={'player_positions': 'Pozycje'}, inplace=True)
        data.rename(columns={'overall': 'Overall'}, inplace=True)
        data.rename(columns={'age': 'Wiek'}, inplace=True)
        return data

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

    roma = pad.read_csv('static/data/roma.csv')
    roma.rename(columns={'actual_value': 'Rzeczywista wartość'}, inplace=True)
    roma.rename(columns={'Model prediction': 'Przewidywana wartość'}, inplace=True)
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

    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Manchester City')
    df['value_eur'] = (df["value_eur"] / 1e6).round(2).astype(str) + " M Euro"

    def rename(data):
        data.rename(columns={'short_name': 'Nazwa zawodnika'}, inplace=True)
        data.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
        data.rename(columns={'value_eur': 'Wartosc'}, inplace=True)
        data.rename(columns={'player_positions': 'Pozycje'}, inplace=True)
        data.rename(columns={'overall': 'Overall'}, inplace=True)
        data.rename(columns={'age': 'Wiek'}, inplace=True)
        return data

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

    mc = pad.read_csv('static\data\mc.csv')
    rename(mc)
    mc = mc.sort_values('Zarobki', ascending=False)

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
    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Legia Warszawa')

    df['value_eur'] = (df["value_eur"] / 1e6).round(2).astype(str) + " M Euro"

    def rename(data):
        data.rename(columns={'short_name': 'Nazwa zawodnika'}, inplace=True)
        data.rename(columns={'wage_eur': 'Zarobki'}, inplace=True)
        data.rename(columns={'value_eur': 'Wartosc'}, inplace=True)
        data.rename(columns={'player_positions': 'Pozycje'}, inplace=True)
        data.rename(columns={'overall': 'Overall'}, inplace=True)
        data.rename(columns={'age': 'Wiek'}, inplace=True)

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

    legia  = pad.read_csv('static\data\legia.csv')
    legia = legia.sort_values('Zarobki', ascending=False)
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
    z3 = data[data['short_name'] == z3][  ['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
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
            'legia' : legia,
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

