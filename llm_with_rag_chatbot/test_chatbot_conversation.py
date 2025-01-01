"""
This script tests a chatbot with a prepared conversation.
The conversation is in JSON format as a list of objects.
Each object has two fields: "query", which is the query given to the chatbot,
and "rubric", an object containing information regarding how to evaluate the
response.
The rubric object has two fields, "standard", which is an example of a
correct output, and "keywords", which is an object containing terms
that should appear in a correct response.
The keywords object has fields "containsAny" and/or "containsAll".
Each of these fields contains a list of strings.
"containsAny" means that the chatbot response must contain any one of
these strings (ignoring case) to be considered correct.
"containsAll" means that the chatbot response must contain ALL of
these strings (ignoring case) to be considered correct.
If both fields are present, both conditions must be met.
"""

import argparse
from collections import deque
from datetime import datetime
import json
import logging
import os
import re

from dotenv import load_dotenv

from evaluators import KeywordEvaluator, RougelEvaluator, LlmEvaluator
from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

TIME_FMT = "%Y-%m-%d-%H%M.%S.%f"
ENC = "utf-8"

start_time = datetime.now()

# Print message so that user does not think program has frozen.
print("Chatbot test running...\n")

# Load environment variables.
load_dotenv()

# Process command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--conversation", "-c", type=str, required=True, help="Text file containing prepared conversation.")
parser.add_argument("--output_log", "-o",
                    default="chatbot_test_output_{}.log".format(start_time.strftime(TIME_FMT)),
                    type=str, help="If desired, specify a file other than the default to which the log the chatbot test output.")
parser.add_argument("--log_level", "-l", default="INFO", type=str, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                    help="If desired, specify a logging level other than the default of INFO.")
args = parser.parse_args()

# Set up logging.
logger = logging.getLogger(__name__)
logging.basicConfig(filename=args.output_log, filemode="w", level=logging.getLevelNamesMapping()[args.log_level], encoding=ENC)

# Store the conversation history.
conversation_history = deque(maxlen=10)

# Initialize Evaluator objects to keep track of correctness.
evaluators = (KeywordEvaluator(), RougelEvaluator(), LlmEvaluator(os.getenv("OPENAI_API_KEY"), os.getenv("EVAL_MODEL")))

# Print model and architecture information.
logger.info("Model and Architecture Information:\n")
logger.info("Architecture: LLM with KO RAG and Code Executor LLM\n")  # TODO: Make this configurable when other architectures are available.
logger.info("Model name: {}\n".format(os.getenv("MODEL")))
logger.info("Model seed: {}\n".format(os.getenv("MODEL_SEED")))
logger.info("RAG Knowledge Base: {}\n".format(os.getenv("KNOWLEDGE_BASE")))
logger.info("\n")

# Keep track of chat transcript for logging purposes.
transcript = []

# Feed the chatbot each query in the conversation and score each resulting response.
with open(args.conversation, mode='r', encoding=ENC) as conversation_file:
    conversation = json.load(conversation_file)
    logger.info("Transcript for Conversation {}:\n".format(args.conversation))
    for exchange in conversation:
        query = exchange["query"]
        transcript.append("USR> {}\n".format(query))
        logger.info(transcript[-1])

        # Feed the chatbot the query.
        response = process(query, conversation_history)
        transcript.append("BOT> {}\n".format(response))
        logger.info(transcript[-1])

        # Score the response using each method of evaluation.
        for e in evaluators:
            e.record_response(response, exchange)

        # Update the conversation history for the chatbot. TODO: Encapsulate this logic in the chatbot itself.
        code = (
            re.search(r"```(.*?)```", response, re.DOTALL).group(1)
            if "```" in response
            else ""
        )
        conversation_history.append(
            (query, response.replace(code, ""))
        )  # update history excluding code

# Report results of chatbot testing.
logger.info("Chat Transcript:\n{}\n".format("\n".join(transcript)))
for e in evaluators:
    logger.info("{}\n".format(e.get_results()))

# Record time elapsed.
end_time = datetime.now()
elapsed = end_time - start_time
logger.info("Ran test in {}\n".format(elapsed))

print("Ran test in {}\n".format(elapsed))
print("Output in {}\n".format(args.output_log))
