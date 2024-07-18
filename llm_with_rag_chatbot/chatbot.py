import os
import threading

from flask import Flask, jsonify, render_template, request

from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


def background_task(user_question, chat_history_tuples):
    bot_response = process(user_question, chat_history_tuples)
    return bot_response


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_question = data["question"]
    chat_history = data["history"]
    chat_history_tuples = [
        (item["question"], item["response"]) for item in chat_history
    ]

    # Start a background thread for the task
    thread = threading.Thread(
        target=background_task, args=(user_question, chat_history_tuples)
    )
    thread.start()
    thread.join()  # Wait for the thread to complete

    # Get the result
    bot_response = background_task(user_question, chat_history_tuples)

    return jsonify(response=bot_response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
