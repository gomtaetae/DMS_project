from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from server import config

db = SQLAlchemy()
migrate = Migrate()

# create_app함수가 app 객체를 생성해 반환하도록 하는 이유는
# app=Flask(__name__)을 전역으로 사용하게되면 오류가 발생하기 때문(순환 참조 오류)
# create_app이 '애플리케이션 팩토리'이다.
# create_app 대신 다른 이름을 사용하면 정상작동하지 않는다.
def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # ORM
    db.init_app(app)
    migrate.init_app(app, db)
    from . import models

    # 블루프린트 작업 : Flask에서 URL과 함수의 매핑을 관리하기 위해 사용하는 도구이다.
    # 블루프린트를 이용하면 라우팅 함수를 체계적으로 관리할 수 있다.
    from .views import main_views, auth_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(auth_views.bp)


    return app