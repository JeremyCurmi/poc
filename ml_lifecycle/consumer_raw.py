import os
import json
import sys
import pika
import requests


QUEUE_NAME = 'raw_data'


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME)

    def callback(ch, method, properties, body):
        print(body)
        payload_str = body.decode("UTF-8").replace("NaN","null")
        print("enriched payload: ", payload_str)

        # send to pipeline api
        r = requests.post('http://0.0.0.0:8000/pipeline', data=payload_str)

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
