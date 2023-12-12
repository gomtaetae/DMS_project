from server.pybo import db
from datetime import datetime

class Users(db.Model):
    __tablename__ = 'Users'

    users_id = db.Column(db.Integer, primary_key=True)
    nickname = db.Column(db.String(60), nullable=False)
    users_name = db.Column(db.String(45), nullable=False)
    email = db.Column(db.String(60), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    users_birth = db.Column(db.Date)
    users_phone = db.Column(db.String(45))
    address_main = db.Column(db.String(255))
    address_sub = db.Column(db.String(255))
    createdate = db.Column(db.DateTime, default=datetime.utcnow)
    updatedate = db.Column(db.TIMESTAMP)


class Sleep(db.Model):
    __tablename__ = 'Sleep'

    sleep_id = db.Column(db.Integer, primary_key=True)
    users_id = db.Column(db.Integer, db.ForeignKey('Users.users_id', ondelete='CASCADE'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    video_url = db.Column(db.String(255))  # 동영상 파일 경로
    capture_url = db.Column(db.String(255))  # 캡처 이미지 파일 경로

    # Users 테이블과 1:N 관계
    users = db.relationship('Users', backref=db.backref('sleeps'))


class Ciga(db.Model):
    __tablename__ = 'Ciga'

    ciga_id = db.Column(db.Integer, primary_key=True)
    users_id = db.Column(db.Integer, db.ForeignKey('Users.users_id', ondelete='CASCADE'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    video_url = db.Column(db.String(255))  # 동영상 파일 경로
    capture_url = db.Column(db.String(255))  # 캡처 이미지 파일 경로

    # Users 테이블과 1:N 관계
    users = db.relationship('Users', backref=db.backref('cigas'))


class Phone(db.Model):
    __tablename__ = 'Phone'

    phone_id = db.Column(db.Integer, primary_key=True)
    users_id = db.Column(db.Integer, db.ForeignKey('Users.users_id', ondelete='CASCADE'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    video_url = db.Column(db.String(255))  # 동영상 파일 경로
    capture_url = db.Column(db.String(255))  # 캡처 이미지 파일 경로

    # Users 테이블과 1:N 관계
    users = db.relationship('Users', backref=db.backref('phones'))