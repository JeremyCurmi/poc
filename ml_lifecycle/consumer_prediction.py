import ast
import os
import sys
import pika
from pymongo import MongoClient

QUEUE_NAME = 'prediction'
client = MongoClient("mongodb://localhost:27017", username='root', password='root')


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME)

    def callback(ch, method, properties, body):
        payload_str = body.decode("UTF-8").replace("null", "None")
        payload = eval(payload_str)
        print(f"prediction data received: {payload}")

        # TODO send msg to database
        prediction_db = client['prediction']
        mycol = prediction_db["prediction"]

        print(f"type of body:{type(payload)}")
        result = mycol.insert_one(payload)
        print(f"Inserting prediction data in db: {payload}")

    channel.basic_consume(
        queue=QUEUE_NAME,
        auto_ack=True,
        on_message_callback=callback
    )

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
