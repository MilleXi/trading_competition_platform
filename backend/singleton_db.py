from flask_sqlalchemy import SQLAlchemy
from singleton import SingletonMeta

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.db = SQLAlchemy()
