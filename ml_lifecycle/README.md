# ml-lifecycle
The entire ML lifecycle POC

```
docker-compose up -d
python db_migrations.py
python consumer_raw.py
python api_pipe.py
python consumer_pipeline.py
python api_prediction.py
python consumer_prediction.py
python producer.py
```