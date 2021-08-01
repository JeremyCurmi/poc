import json
import pika
import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from utils import NpEncoder

QUEUE_NAME = 'prediction'


class ModelPayload(BaseModel):
    pipeline: str
    model: str
    data: Dict[str, Any] = None
    transformed_data: List[dict] = None
    index: Any


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue=QUEUE_NAME)

app = FastAPI()


@app.post("/model/")
async def run_model(payload: ModelPayload):
    with open(payload.model, 'rb') as file:
        pkl_model = pickle.load(file)
    example = pd.DataFrame(payload.transformed_data)
    prediction = pkl_model.predict(example)
    print(f"example and prediction: {example}, {prediction}")
    enriched_payload = payload.__dict__
    enriched_payload['prediction'] = prediction.tolist()

    print(enriched_payload)
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(enriched_payload, indent=2, cls=NpEncoder).encode('utf-8')
    )

    return enriched_payload


if __name__ == "__main__":
    uvicorn.run("api_prediction:app", host="0.0.0.0", port=8080, reload=True, debug=True)