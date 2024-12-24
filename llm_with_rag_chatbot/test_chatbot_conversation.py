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
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from rouge import Rouge

from llm_with_rag_chatbot.openai_chatbot_with_assistant_api import process

TIME_FMT = "%Y-%M-%d-%H%M.%S.%f"
ENC = "utf-8"

class Evaluator:
    """
    Evaluates chatbot responses.
    """
    def __init__(self):
        self._scores = []

    def get_name(self) -> str:
        """
        Returns the name of the approach/metric used by this Evaluator.
        """
        pass

    def score(self, response: str, exchange: dict) -> float:
        """
        Scores the chatbot response and makes an internal note of its correctness.

        :param response: the chatbot response
        :param exchange: a dict containing the query given to the chatbot and a rubric with which to evaluate the chatbot's response
        """
        pass

    def record_response(self, response: str, exchange: dict) -> None:
        """
        Scores the chatbot response and makes an internal note of its correctness.

        :param response: the chatbot response
        :param exchange: a dict containing the query given to the chatbot and a rubric with which to evaluate the chatbot's response
        """
        self._scores.append(self.score(response, exchange))

    def get_results(self) -> str:
        """
        Returns results of evaluating all chatbot responses.
        """
        points = sum(self._scores)
        total = len(self._scores)
        return "{} Evaluation Results:\nTotal score {}\n{} points out of {}\nIndividual items: {}".format(self.get_name(), points / total, points, total, self._scores)

class KeywordEvaluator(Evaluator):
    """
    Evaluates chatbot responses based on the presence of keywords specified in the rubric.
    See the file-level docstring for documentation of the "keywords" field of the rubric.
    """
    def score(self, response: str, exchange: dict) -> float:
        keywords = exchange["rubric"]["keywords"]
        for rule in keywords:
            if rule == "containsAll":
                if not all(term.casefold() in response.casefold() for term in keywords[rule]):
                    return 0.0
            elif rule == "containsAny":
                if not any(term.casefold() in response.casefold() for term in keywords[rule]):
                    return 0.0
            else:
                raise ValueError("Invalid field {} found in keywords!".format(rule))
        return 1.0
    
    def get_name(self) -> str:
        return "Keyword"

class RougelEvaluator(Evaluator):
    """
    Evaluates each response by computing the ROUGE-L score between it and the standard.
    """
    def score(self, response: str, exchange: dict) -> float:
        rouge = Rouge(metrics=["rouge-l"], stats=["f"])
        return rouge.get_scores(response, exchange["rubric"]["standard"])[0]["rouge-l"]["f"]
    
    def get_name(self) -> str:
        return "ROUGE-L"

class LlmEvaluator(Evaluator):
    """
    Uses an LLM to assess how close the chatbot's responses are to gold standard responses.
    """
    def score(self, response: str, exchange: dict) -> float:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=os.getenv("MODEL"))
        prompt = PromptTemplate.from_template(
            """You are grading an agent's responses to questions.
            Question: {question}
            Example of correct response: {standard}
            Agent's response: {response}
            The agent's response is considered correct if it has the same meaning as the example response, even if it is not identical.
            Is the agent's response correct?
            Please answer CORRECT or INCORRECT.
            """
        )
        fields = {
            "question": exchange["query"],
            "standard": exchange["rubric"]["standard"],
            "response": response
        }
        result = model.invoke(prompt.invoke(fields))
        if result.content.lower() == "correct":
            return 1.0
        else:
            return 0.0
        
    def get_name(self) -> str:
        return "LLM"

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
evaluators = (KeywordEvaluator(), RougelEvaluator(), LlmEvaluator())

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
