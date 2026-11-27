import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Replaces the MySQL connection string
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'a533ad956c07e00564024561e99a0b35'