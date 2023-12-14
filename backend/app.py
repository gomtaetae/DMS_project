from flask import Flask, jsonify, request, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv

from my_model import my_model
from my_util.my_logger import my_logger

import os, traceback

load_dotenv()
# instantiate the app
app = Flask(__name__)
app.secret_key = 'laksdjfoiawjewfansldkfnzcvjlzskdf'

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = os.getenv("SQLALCHEMY_TRACK_MODIFICATIONS")
db = SQLAlchemy()
db.init_app(app)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

bcrypt = Bcrypt(app)

# sanity check route
@app.route('/', methods=['GET'])
def test_router():
    my_logger.info("hello this is root url!")
    return jsonify('This is Docker Test developments Server!')

@app.route('/health_check', methods=['GET'])
def health_check():
    my_logger.info("health check route url!")
    return jsonify('good')

# 회원가입
# 회원가입
@app.route("/api/auth/signup", methods=['POST'])
def auth_signup():
    my_logger.info("SignUp!")
    data = request.get_json()
    my_logger.info("Received data: %s", data)

    # 입력 데이터 검증
    if not all(key in data for key in ['nickname', 'users_name', 'email', 'password']):
        return jsonify({'fail': 'Missing required fields.'}), 400

    nickname = data.get('nickname')
    users_name = data.get('users_name')
    email = data.get('email')
    password = data.get('password')
    if not password:
        return jsonify({'fail': 'Password must be provided.'}), 400
    users_birth = data.get('users_birth')
    users_phone = data.get('users_phone')
    address_main = data.get('address_main')
    address_sub = data.get('address_sub')

    # 닉네임 중복 확인
    if my_model.User.query.filter_by(nickname=nickname).first():
        return jsonify({'fail': 'Nickname is already in use.'}), 409

    # 이메일 중복 확인
    if my_model.User.query.filter_by(email=email).first():
        return jsonify({'fail': 'Email is already in use.'}), 409

    try:
        # 비밀번호 해싱
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = my_model.User(
            nickname=nickname,
            users_name=users_name,
            email=email,
            password=hashed_password,
            users_birth=users_birth,
            users_phone=users_phone,
            address_main=address_main,
            address_sub=address_sub
        )
        my_model.db.session.add(user)
        my_model.db.session.commit()
        my_logger.info("User Save Success")
        return jsonify({'success': 'User registration successful.'})
    except Exception as e:
        my_logger.error("User Save Fail: %s", str(e))
        error_info = traceback.print_exc()  # 예외의 상세한 정보를 출력
        print("트레이스백:", error_info)
        return jsonify({'fail': 'An error occurred during signup. Please try again later.'}), 500

# 로그인
@app.route("/api/auth/signin", methods=['POST'])
def auth_login():
    my_logger.info("User auth Login")
    my_logger.info(request.get_json())

    email = request.get_json().get('email')
    password = request.get_json()['password']

    my_user = my_model.User()

    try:
        user_data = my_user.query.filter_by(email=email).first()

        if user_data is not None:
            my_logger.info(user_data.password)
            auth = user_data.check_password(password)
            if not auth:
                my_logger.info("password validation fail!")
                return jsonify({'error': 'Invalid password'}), 401
            else:
                my_logger.info("login success!")
                # 토큰 생성 및 반환 로직을 여기에 추가
                return jsonify({'success': 'User login successful.'})
        else:
            my_logger.info("User not found or incorrect credentials.")
            return jsonify({'error': 'User not found or incorrect credentials.'}), 404
    except Exception as e:
        my_logger.error("Login exception: %s", str(e))
        # 디버그 정보는 로그로만 남기고 사용자에게는 노출하면 안된다
        return jsonify({'error': 'An internal error occurred.'}), 500

    @app.route('/api/auth/logout')
    def auth_logout():
        session['login'] = False
        return {'success': 'logout'}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 데이터베이스 테이블 생성
    app.run(host='0.0.0.0',port=os.getenv('FLASK_RUN_PORT'),debug=os.getenv('FLASK_DEBUG'))