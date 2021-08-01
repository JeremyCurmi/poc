import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017", username='root', password='root')
data_db = client['data']
prediction_db = client['prediction']


class ValidationPayload(BaseModel):
    index: int


app = FastAPI()


@app.get("/validate/{index}")
async def run_monitor(index: int):
    actual = data_db.collection.find_one({'PassengerId': index}, {'Survived'})
    prediction = prediction_db.prediction.find_one({'index': index}, {'prediction'})

    return {
        "actual":actual['Survived'],
        "prediction":prediction['prediction'][0]
    }


if __name__ == "__main__":
    uvicorn.run("api_monitor:app", host="0.0.0.0", port=8002, reload=True, debug=True)
