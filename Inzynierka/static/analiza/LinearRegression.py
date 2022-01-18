import plotly
import pandas as pad
import seaborn as sea
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
pad.set_option('display.max_rows',None)
pad.set_option('display.max_columns',None)
pad.set_option('display.width',None)
import sqlite3
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


df = pad.read_csv(r'D:\Studia\Praca inżynierska\players_20.csv')
# mu= df[df.club == "Manchester United"]
# parsing the DataFrame in json format.
json_records = df.reset_index().to_json(orient='records')
print(df)
