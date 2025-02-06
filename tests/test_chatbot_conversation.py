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

from dotenv import load_dotenv

from evaluators import KeywordEvaluator, BleuEvaluator, Rouge1Evaluator, Rouge2Evaluator, RougeLEvaluator, LlmEvaluator
from src.chatbots import init_chatbot_from_str

TIME_FMT = "%Y-%m-%d-%H%M.%S.%f"
ENC = "utf-8"

def select(conversation: list, indices: list) -> list:
    if indices is None:
        return conversation
    else:
        for i in indices:
            if (i < 0) or i >= len(conversation):
                raise ValueError("Invalid test case {}".format(i))
        return (conversation[i] for i in indices)

start_time = datetime.now()

# Print message so that user does not think program has frozen.
print("Chatbot test running...\n")

# Load environment variables.
load_dotenv()

# Process command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--chatbot_architecture", "-a", type=str, choices=("LlmWithRagKosAndExternalInterpreter", "LlmWithKoCodeTools", "PlainLlm", "LlmWithKoRagMetadataAndCodeTools"), required=True, help="Which chatbot architecture to test.")
parser.add_argument("--conversation", "-c", type=str, required=True, help="Text file containing prepared conversation.")
parser.add_argument("--test_cases", "-t", nargs="+", type=int, help="If desired, specify which test cases to run. The first is #0. Default is to run all tests.")
parser.add_argument("--output_log", "-o",
                    default="chatbot_test_output_{}.log".format(start_time.strftime(TIME_FMT)),
                    type=str, help="If desired, specify a file other than the default to which the log the chatbot test output.")
parser.add_argument("--log_level", "-l", default="INFO", type=str, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                    help="If desired, specify a logging level other than the default of INFO.")
args = parser.parse_args()

# Set up logging.
logger = logging.getLogger(__name__)
logging.basicConfig(filename=args.output_log, filemode="w", level=logging.getLevelNamesMapping()[args.log_level], encoding=ENC)

# Initialize Evaluator objects to keep track of correctness.
evaluators = (KeywordEvaluator(), BleuEvaluator(), Rouge1Evaluator(), Rouge2Evaluator(), RougeLEvaluator(), LlmEvaluator(os.getenv("OPENAI_API_KEY"), os.getenv("EVAL_MODEL")))

# Initialize chatbot.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL")
knowledge_base = os.getenv("KNOWLEDGE_BASE")
model_seed = int(os.getenv("MODEL_SEED"))
embedding_model = os.getenv("EMBEDDING_MODEL")
embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION"))
chatbot = init_chatbot_from_str(args.chatbot_architecture, OPENAI_API_KEY, model_name, model_seed, knowledge_base, embedding_model, embedding_dimension)

# Print model and architecture information.
logger.info("Model and Architecture Information:\n")
logger.info("Architecture: {})\n".format(chatbot.get_architecture()))
logger.info("Model name: {}\n".format(model_name))
logger.info("Model seed: {}\n".format(model_seed))
logger.info("RAG Knowledge Base: {}\n".format(knowledge_base))
logger.info("\n")

# Keep track of chat transcript for logging purposes.
transcript = []

# Feed the chatbot each query in the conversation and score each resulting response.
with open(args.conversation, mode='r', encoding=ENC) as conversation_file:
    conversation = json.load(conversation_file)

    # Select a subset of the conversation if desired.
    selected_conversation = select(conversation, args.test_cases)

    for exchange in selected_conversation:
        query = exchange["query"]
        transcript.append("USR> {}\n".format(query))
        logger.info(transcript[-1])

        # Feed the chatbot the query.
        response = chatbot.invoke(query)
        transcript.append("BOT> {}\n".format(response))
        logger.info(transcript[-1])

        # Score the response using each method of evaluation.
        for e in evaluators:
            e.record_response(response, exchange)

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
