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


df = pad.read_csv("test.csv")
# df = df.DataFrame.to_html(classes='mystyle')
print(df)
