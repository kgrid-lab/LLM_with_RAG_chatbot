"""
This script tests a chatbot with a prepared conversation.
The conversation is in JSON format as a list of objects.
Each object has two fields: the query, which is the query given to the chatbot,
and the rubric, an object containing information regarding how to evaluate the
response.
The rubric object has two fields, the standard, which is an example of a
correct output, and the keywords, which is an object containing terms
that should appear in a correct response.
Documentation of the keywords object is found in the score function.
"""

import argparse
from collections import deque
from datetime import datetime
import json
import os
import re

from dotenv import load_dotenv

from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

TIME_FMT = "%Y-%M-%d-%H%M.%S.%f"
ENC = "utf-8"

def score(response: str, keywords: dict) -> bool:
    """
    Scores the chatbot response according to the list of keywords.
    Returns True if the response is correct and False otherwise.
    The keywords is a dict with keys representing rules and values representing terms.
    For the response to be considered correct, all rules must be satisfied.
    The "containsAll" rule is satisfied if all the terms are contained in the response.
    The "containsAny" rule is satisfied if any of the terms are contained in the response.
    Case is ignored.
    """
    for rule in keywords:
        if rule == "containsAll":
            if not all(term.casefold() in response.casefold() for term in keywords[rule]):
                return False
        if rule == "containsAny":
            if not any(term.casefold() in response.casefold() for term in keywords[rule]):
                return False
    return True

def report(msg, f):
    """
    Writes msg to both stdout and the file f.
    TODO: Find more "pythonic" way of doing this.
    """
    print(msg)
    f.write(msg)

start_time = datetime.now()

# Process command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--conversation", "-c", type=str, required=True, help="Text file containing prepared conversation.")
parser.add_argument("--output_log", "-o",
                    default="chatbot_output_{}.log".format(start_time.strftime(TIME_FMT)),
                    type=str, help="If desired, specify a file other than the default to which the log the chatbot output.")
parser.add_argument("--result_log", "-r",
                    default="result_{}.log".format(start_time.strftime(TIME_FMT)),
                    type=str, help="If desired, specify a file other than the default to which the log the test results.")
args = parser.parse_args()

# Store the conversation history.
conversation_history = deque(maxlen=10)

# Keep track of correctness.
items = []

# Feed the chatbot each query in the conversation and score each resulting response.
# Meanwhile, write the chatbot responses to the provided file for debugging purposes.
with open(args.conversation, mode='r', encoding=ENC) as conversation_file:
    with open(args.output_log, mode='w', encoding=ENC) as response_output:
        conversation = json.load(conversation_file)
        for exchange in conversation:
            response = process(exchange["query"], conversation_history)
            response_output.write(response)
            if score(response, exchange["rubric"]["keywords"]):
                items.append(1)
            else:
                items.append(0)
            code = (
                re.search(r"```(.*?)```", response, re.DOTALL).group(1)
                if "```" in response
                else ""
            )
            conversation_history.append(
                (exchange["query"], response.replace(code, ""))
            )  # update history excluding code

# Report results to stdout and log file.
with open(args.result_log,  mode='w', encoding=ENC) as result_log_file:
    # Report model and architecture information.
    report("model: {}".format(os.getenv("MODEL")), result_log_file)
    report("knowledge_base: {}".format(os.getenv("KNOWLEDGE_BASE")), result_log_file)
    report("model_seed: {}".format(os.getenv("MODEL_SEED")), result_log_file)

    # Report score information.
    points = sum(items)
    total = len(items)
    report("Total score {}".format(points / total), result_log_file)
    report("{} points out of {}".format(points, total), result_log_file)
    report("Individual items: {}".format(items), result_log_file)

    end_time = datetime.now()
    report("Ran test in {}".format(end_time - start_time), result_log_file)
