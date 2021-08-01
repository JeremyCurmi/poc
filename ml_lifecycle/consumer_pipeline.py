import os
import json
import sys
import pika
import requests

QUEUE_NAME = 'processed_data'


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME)

    def callback(ch, method, properties, body):

        payload_str = body.decode("UTF-8").replace("NaN", "null")
        print(f"processed data received: {payload_str}")

        # send to model api
        r = requests.post('http://0.0.0.0:8080/model', data=payload_str)

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
