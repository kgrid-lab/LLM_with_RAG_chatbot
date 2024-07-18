import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request
from flask_caching import Cache

from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE": "simple"})


@app.route("/")
def home():
    return render_template("index.html")


def background_task(task_id, user_question, chat_history_tuples):
    response = process(user_question, chat_history_tuples)
    cache.set(task_id, response)


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_question = data["question"]
    chat_history = data["history"]
    chat_history_tuples = [
        (item["question"], item["response"]) for item in chat_history
    ]

    task_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=background_task, args=(task_id, user_question, chat_history_tuples)
    )
    thread.start()

    return jsonify(task_id=task_id)


@app.route("/check_response/<task_id>", methods=["GET"])
def check_response(task_id):
    response = cache.get(task_id)
    if response:
        return jsonify(response=response)
    else:
        return jsonify(status="processing")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
