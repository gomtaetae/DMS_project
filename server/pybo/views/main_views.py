from flask import Blueprint, render_template

bp = Blueprint('main', __name__, url_prefix='/')

# 이 작업을 통해서 라우팅 함수는 main_views 파일에 작업하면된다.

@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo!'

@bp.route('/')
def index():
    return render_template('main.html')