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
    df = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
    data = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Roma')

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
        }
    )


def manchester(request):
    df = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
    data = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')

    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Manchester City')

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
        }
    )


def legia(request):
    df = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
    data = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
    def club(nazwa):
        return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]
    df = club('Legia Warszawa')

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

        return longlist.sort_values("Zarobki").head(1)

    rename(df)

    x=df.nlargest(3, 'Zarobki')
    z = x.iloc[0]

    z=z['Nazwa zawodnika']


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
        }
    )


def general(request):
    # df = pad.read_csv(r'static/data.csv')
    # df = df.drop('Unnamed: 0', 1)
    # data = df
    # data = data.to_html

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
    df = pad.read_csv("static/test.csv")
    html = df.to_html()
    html = {'html': html}
    return render(request, 'Dane/clubs.html', html)


# def club_analisys(request):
#     df = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
#
#     def club(nazwa):
#         return df[df['club'] == nazwa][['short_name', 'wage_eur', 'value_eur', 'player_positions', 'overall', 'age']]


# def test(request):
#     df = pad.read_csv("test.csv")
#
#     context = {
#         'test': df
#     }
#     return render(request, 'Dane/index.html', context)
# def mu(request):
#
#     """ Stworzenie data frame'u zawierającego statystyki zawodników z gry Fifa 20"""
#     df = pad.read_csv("static/players_20.csv")
#     mu= df[df.club == "Manchester United"]
#     # parsing the DataFrame in json format.
#     json_records = mu.reset_index().to_json(orient='records')
#     data = []
#     data = json.loads(json_records)
#     contextt = {'d': data}
#
#     return render(request, 'Dane/clubs.html', contextt)
# def test(request):
#
#     """ Stworzenie data frame'u zawierającego statystyki zawodników z gry Fifa 20"""
#     df = pad.read_csv('Users/mbozik/PycharmProjects/Inzynierka/Inzynierka/static/test.csv')
#     # mu= df[df.club == "Manchester United"]
#     # parsing the DataFrame in json format.
#     json_records = df.reset_index().to_json(orient='records')
#     data = []
#     data = json.loads(json_records)
#     contextt = {'d': data}
#
#     return render(request, 'Dane/clubs.html', contextt)

# def Table(request):
#     df = pad.read_csv("static/test.csv")
#
#     geeks_object = df.to_html()
#
#     return HttpResponse(geeks_object)
# def table(request):
#     df = pad.read_csv("test.csv")
#     test = df.to_html()
#     return render(request, 'Inzynierka/Dane/templates/Dane/clubs.html', {'table': test})

# def test(request):
#     df = pad.DataFrame(numpy.random(1,10,(5,3)),columns=["A","B","C"])
#     test = df.to_html()
#     return render(request, 'Inzynierka/Dane/templates/Dane/clubs.html', {'table': test})
