# coding: utf-8
from flask_bcrypt import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()



class Ciga(db.Model):
    __tablename__ = 'Ciga'

    ciga_id = db.Column(db.Integer, primary_key=True)
    users_id = db.Column(db.ForeignKey('Users.users_id', ondelete='CASCADE'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    video_url = db.Column(db.String(255))
    capture_url = db.Column(db.String(255))

    users = db.relationship('User', primaryjoin='Ciga.users_id == User.users_id', backref='cigas')



class Phone(db.Model):
    __tablename__ = 'Phone'

    phone_id = db.Column(db.Integer, primary_key=True)
    users_id = db.Column(db.ForeignKey('Users.users_id', ondelete='CASCADE'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    video_url = db.Column(db.String(255))
    capture_url = db.Column(db.String(255))

    users = db.relationship('User', primaryjoin='Phone.users_id == User.users_id', backref='phones')



class Sleep(db.Model):
    __tablename__ = 'Sleep'

    sleep_id = db.Column(db.Integer, primary_key=True)
    users_id = db.Column(db.ForeignKey('Users.users_id', ondelete='CASCADE'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    video_url = db.Column(db.String(255))
    capture_url = db.Column(db.String(255))

    users = db.relationship('User', primaryjoin='Sleep.users_id == User.users_id', backref='sleeps')



class User(db.Model):
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
    createdate = db.Column(db.DateTime, server_default=db.FetchedValue())
    updatedate = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    def has_password(self):
        self.password = generate_password_hash(self.password).decode('utf8')

    def check_password(self, password):
        return check_password_hash(self.password, password)