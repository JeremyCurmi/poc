import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017",username='root', password='root')

# delete database
client.drop_database('prediction')

# create database
prediction_db = client['prediction']

# delete data database
client.drop_database('prediction')

# create data database
data_db = client['data']

# upload data
data_db.collection.insert_many(pd.read_csv('data/snapshots/train.csv').to_dict(orient='records'))

