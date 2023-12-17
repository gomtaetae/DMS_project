@echo off
cd C:\Users\KOSA\Downloads\DMS_project\server

call C:\Users\KOSA\anaconda3\envs\DMS\Lib\venv\scripts\nt\activate

set FLASK_APP=pybo
set FLASK_DEBUG=true

flask db migrate
flask db upgrade

flask run
flask-sqlacodegen "mysql+pymysql://root:1234@db/dms" --flask > model.py
flask-sqlacodegen "mysql+pymysql://root:1234@db/dms" --flask
pause