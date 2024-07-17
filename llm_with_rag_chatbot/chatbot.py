from flask import Flask, jsonify, render_template, request

from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_question = data["question"]
    chat_history = data["history"]
    chat_history_tuples = [
        (item["question"], item["response"]) for item in chat_history
    ]
    # Get response from the chatbot
    bot_response = process(user_question, chat_history_tuples)
    return jsonify(response=bot_response)


if __name__ == "__main__":
    app.run(debug=True)
