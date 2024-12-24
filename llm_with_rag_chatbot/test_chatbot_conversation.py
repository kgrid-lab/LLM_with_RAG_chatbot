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
import os
import re

from dotenv import load_dotenv

from evaluators import KeywordEvaluator, RougelEvaluator, LlmEvaluator
from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

TIME_FMT = "%Y-%m-%d-%H%M.%S.%f"
ENC = "utf-8"

def report(msg, f):
    """
    Writes msg to both stdout and the file f.
    TODO: Find more "pythonic" way of doing this.
    """
    print(msg)
    f.write(msg)

start_time = datetime.now()

# Load environment variables
load_dotenv()

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
evaluators = (KeywordEvaluator(), RougelEvaluator(), LlmEvaluator(os.getenv("OPENAI_API_KEY"), os.getenv("MODEL")))

# Feed the chatbot each query in the conversation and score each resulting response.
# Meanwhile, write the chatbot responses to the provided file for debugging purposes.
with open(args.conversation, mode='r', encoding=ENC) as conversation_file:
    with open(args.output_log, mode='w', encoding=ENC) as response_output:
        conversation = json.load(conversation_file)
        for exchange in conversation:
            response = process(exchange["query"], conversation_history)
            response_output.write(response)
            for e in evaluators:
                e.record_response(response, exchange)
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
    for e in evaluators:
        report(e.get_results(), result_log_file)

    end_time = datetime.now()
    report("Ran test in {}".format(end_time - start_time), result_log_file)
