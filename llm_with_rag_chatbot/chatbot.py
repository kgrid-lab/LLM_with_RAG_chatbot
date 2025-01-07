import os
import threading
import uuid
import logging
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_caching import Cache

from chatbots import LlmWithRagKosAndExternalInterpreter

# Configure logging to include date and time
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define format for logs
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

# File handler
file_handler = RotatingFileHandler('chatbot_logs.log', maxBytes=10000, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler for Heroku logs
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE": "simple"})

# Initialize the chatbot.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL")
knowledge_base = os.getenv("KNOWLEDGE_BASE")
model_seed = int(os.getenv("MODEL_SEED"))
chatbot = LlmWithRagKosAndExternalInterpreter(OPENAI_API_KEY, model_name, model_seed, knowledge_base)

@app.route("/")
def home():
    return render_template("index.html")


def background_task(task_id, user_question):
    response = chatbot.invoke(user_question)
    # Log the question and response with date and time
    logger.info("Question: %s", user_question)
    logger.info("Response: %s", response)
    cache.set(task_id, response)


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_question = data["question"]

    task_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=background_task, args=(task_id, user_question)
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