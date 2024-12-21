"""
This script tests a chatbot with a prepared conversation.
Subjects the chosen chatbot to the chosen conversation and checks each response against a set of given answers.
Outputs performance information (i.e. score) to the command line.
"""

import argparse
from collections import deque
import json
import re

from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

def score(response: str, rubric: dict) -> bool:
    """
    Scores the chatbot response according to the rubric.
    Returns True if the response is correct and False otherwise.
    The rubric is a dict with keys representing rules and values representing terms.
    For the response to be considered correct, all rules must be satisfied.
    The "containsAll" rule is satisfied if all the terms are contained in the response.
    The "containsAny" rule is satisfied if any of the terms are contained in the response.
    Case is ignored.
    """
    for rule in rubric:
        if rule == "containsAll":
            if not all(term.casefold() in response.casefold() for term in rubric[rule]):
                return False
        if rule == "containsAny":
            if not any(term.casefold() in response.casefold() for term in rubric[rule]):
                return False
    return True


# Process command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--conversation", "-c", type=str, required=True, help="Text file containing prepared conversation.")
parser.add_argument("--response_output", "-r", type=str, required=True, help="File to which to write the chatbot responses.")
args = parser.parse_args()

# Store the conversation history.
conversation_history = deque(maxlen=10)

# Keep track of correctness.
items = []

# Feed the chatbot each query in the conversation and score each resulting response.
# Meanwhile, write the chatbot responses to the provided file for debugging purposes.
with open(args.conversation, mode='r', encoding="utf-8") as conversation_file:
    with open(args.response_output, mode='w', encoding="utf-8") as response_output:
        conversation = json.load(conversation_file)
        for exchange in conversation:
            response = process(exchange["query"], conversation_history)
            response_output.write(response)
            if score(response, exchange["rubric"]):
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

# Report the correctness score.
points = sum(items)
total = len(items)
print("Total score {}".format(points / total))
print("{} points out of {}".format(points, total))
print("Individual items: {}".format(items))
