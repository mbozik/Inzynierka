import csv
import datetime

from django.http import HttpResponse
from django.shortcuts import render
import json
import plotly

"""import pyodbc as odbc"""
import pandas as pad
import seaborn as sea
import matplotlib
import numpy as numpy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import sqlite3
from IPython.display import HTML\

from django.shortcuts import render

# Create your views here.

def home(request):
    return render(
        request,
        'Dane/index.html',
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

def general(request):
    return render(
        request,
        'Dane/general.html',
        {
            'title': "General",
        }
    )

def dashboard(request):
    return render(
        request,
        'Dane/dashboard.html',
        {
            'title': "Dashboard",
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

def table(request):
    df = pad.read_csv("static/test.csv")
    html_table = df.to_html(index=False)
    return render('Dane/clubs.html', {'html_table': html_table}, request)


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
