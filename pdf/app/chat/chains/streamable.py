from flask import current_app
from queue import Queue
from threading import Thread
from app.chat.callbacks.stream import StreamingHandler
import time

class StreamableChain:
  def stream(self, input):
    queue = Queue()
    handler = StreamingHandler(queue)
    def task(app_context):
      app_context.push() #for only flask
      self(input, callbacks=[handler])
    Thread(target=task, args=[current_app.app_context()]).start()
    while True:
      time.sleep(0.1)
      token = queue.get()
      if token is None:
        break
      yield token