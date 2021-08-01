import sys
import json
import pika
import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from utils import NpEncoder

sys.path.append('train')


QUEUE_NAME = 'processed_data'


class PipelinePayload(BaseModel):
    pipeline: str
    model: str
    data: Dict[str, Any] = None
    index: Any


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue=QUEUE_NAME)

app = FastAPI()


@app.post("/pipeline/")
async def run_pipeline(payload: PipelinePayload):
    print("payload: ", payload)
    with open(payload.pipeline, 'rb') as file:
        pkl_pipe = pickle.load(file)
    example = pd.DataFrame.from_dict(payload.data, orient='index').T
    transformed_example = pkl_pipe.transform(example)

    print(f"transformed_example: {transformed_example}")
    enriched_payload = payload.__dict__
    enriched_payload['transformed_data'] = transformed_example.to_dict(orient='records')
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(enriched_payload, indent=2, cls=NpEncoder).encode('utf-8')
    )
    return enriched_payload


if __name__ == "__main__":
    uvicorn.run("api_pipe:app", host="0.0.0.0", port=8000, reload=True, debug=True)