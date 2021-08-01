import json
import pika
import time
import pandas as pd
from utils import NpEncoder

QUEUE_NAME = 'raw_data'
DATA_PATH = 'data/operator/train.csv'
PIPELINE = 'data/artifacts/preprocessing_pipeline.sav'
MODEL = 'data/artifacts/model.sav'
TARGET = 'Survived'
INDEX = 'PassengerId'

df = pd.read_csv(DATA_PATH)
X = df.drop(TARGET,axis=1)
y = df[TARGET]

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue=QUEUE_NAME)

# send message
i = 0

payload = {
    "pipeline": PIPELINE,
    "model": MODEL,
    "data": None,
    "actual": None,
    "index": None
}

while i < df.shape[0]:
    payload['data'] = dict(X.iloc[i])
    payload['actual'] = y[i]
    payload['index'] = X.at[i,INDEX]
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(payload, indent=2, cls=NpEncoder).encode('utf-8')
    )
    # time.sleep(2)
    i += 1

connection.close()
