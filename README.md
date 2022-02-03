## Aplikacja stworzona na potrzeby Pracy Dyplomowej - Big Data w nowoczesnym futbolu
Aplikacja została stworzona w zgodzie z zasadami tworzenia projektów przy pomocy frameworka Django.
Kręgosłup projektu został stworzony dzięki komendzie django-admin startproject, która odpowiada za utworzenie katologu z główną częścią projektu.
Cały projekt jest podzielony na mniejsze podfoldery, z których każdy odpowiada za inny proces generowania aplikacji.

Folder static jest to zbiorczy plik, który zawiera główne elementy statyczne aplikacji.

Folder Inzynierka składa się z plików tworzonych automatycznie podczas generowania projektu przy pomocy Django. 
Z ważniejszych znajduje się tam plik settings, który odpowiada za całą konfigurację instalacji. 
Zawiera on zmienne, które decydują o działaniu całego projektu.
W folderze Dane znajdują się główne pliki obsługujące front-end’owe działanie
aplikacji.


W celu uruchomienia aplikacji należy użyć komendy: python manage.py runserver 
