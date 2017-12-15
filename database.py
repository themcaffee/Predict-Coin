import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///database.db')
Base = declarative_base()


class DataPoint(Base):
    __tablename__ = 'datapoints'
    id = Column(Integer, primary_key=True)
    time = Column(Integer, unique=True)
    low = Column(Integer)
    high = Column(Integer)
    open = Column(Integer)
    close = Column(Integer)
    volume = Column(Integer)


if not os.path.isfile('database.db'):
    Base.metadata.create_all(engine)

session_maker = sessionmaker(bind=engine)
session = session_maker()


def bulk_save_datapoints(points):
    mappings = []
    for point in points:
        mappings.append(dict(time=point[0], low=point[1], high=point[2], open=point[3], close=point[4], volume=point[5]))
    session.bulk_insert_mappings(DataPoint, mappings)


def save_datapoint(time, low, high, open, close, volume):
    point = DataPoint(time=time, low=low, high=high, open=open, close=close, volume=volume)
    session.add(point)
    session.commit()


