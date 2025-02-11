"""
This script tests a chatbot with a prepared conversation.
The conversation is in JSON format as a list of objects.
Each object has two fields: "query", which is the query given to the chatbot,
and "rubric", an object containing information regarding how to evaluate the
response.
The rubric object has two fields, "standard", which is an example of a
correct output, and "keywords", which is an list containing terms
that should appear in a correct response. Comparison happens
in lower case. If there is flexibility regarding which term should appear,
the list entry is an object containing an "any" field listing the options
rather than the term itself. If the term is a number, rounding error
is accounted for in the comparison.
"""

import argparse
from collections import deque
from datetime import datetime
import json
import logging
from math import ceil
import os
import requests

from dotenv import load_dotenv

from evaluators import KeywordEvaluator, BleuEvaluator, Rouge1Evaluator, Rouge2Evaluator, RougeLEvaluator, LlmEvaluator
from src.chatbots import init_chatbot_from_str

TIME_FMT = "%Y-%m-%d-%H%M.%S.%f"
ENC = "utf-8"

def select(conversation: list, indices: list) -> list:
    for i in indices:
        if (i < 0) or i >= len(conversation):
            raise ValueError("Invalid test case {}".format(i))
    return [conversation[i] for i in indices]

def get_openai_usage(openai_admin_key: str, start_time: datetime, end_time: datetime, url: str) -> list[dict]:
    headers = {
        "Authorization": f"Bearer {openai_admin_key}",
        "Content-Type": "application/json"
    }

    duration = end_time - start_time
    num_minutes = ceil(duration.total_seconds() / 60)

    # Define parameters with placeholders for all possible options
    params = {
        "start_time": int(start_time.timestamp()),  # Required: Start time (Unix seconds)
        # "end_time": int(end_time.timestamp()),  # Optional: End time (Unix seconds)
        "bucket_width": "1m",      # Optional: '1m', '1h', or '1d' (default '1d')
        # "project_ids": ["proj_example"],  # Optional: List of project IDs
        # "group_by": ["model"],     # Optional: Fields to group by
        "limit": num_minutes,                 # Optional: Number of buckets to return (default is 7)
        # "page": "cursor_string"   # Optional: Cursor for pagination
    }

    # Initialize an empty list to store all data
    all_data = []

    # Initialize pagination cursor
    page_cursor = None

    # Loop to handle pagination
    while True:
        if page_cursor:
            params["page"] = page_cursor

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data_json = response.json()
            all_data.extend(data_json.get("data", [])) 

            page_cursor = data_json.get("next_page")
            if not page_cursor:
                break  
        else:
            logger.error(f"HTTP error {response.status_code} in response {response} from OpenAI API endpoint {url}")
            break  

    if not all_data:
        logger.error("No data available to retrieve.")

    return all_data

def get_cost(openai_admin_key: str, start_time: datetime, end_time: datetime) -> float:
    """Gets the OpenAI API usage cost in US dollars from start_time up to end_time."""

    # URL of OpenAI Admin Cost API endpoint.
    url = "https://api.openai.com/v1/organization/costs"

    all_data = get_openai_usage(openai_admin_key, start_time, end_time, url)
    return sum(sum(result["amount"]["value"] for result in data["results"]) for data in all_data)

def get_token_usage(openai_admin_key: str, start_time: datetime, end_time: datetime) -> dict:
    """Gets the OpenAI API usage cost in US dollars from start_time up to end_time."""

    # URL of OpenAI Admin Completions Usage API endpoint.
    chat_url = "https://api.openai.com/v1/organization/usage/completions"

    comp_data = get_openai_usage(openai_admin_key, start_time, end_time, chat_url)
    completions = {
        "input_tokens": sum(sum(result["input_tokens"] for result in data["results"]) for data in comp_data),
        "input_cached_tokens": sum(sum(result["input_cached_tokens"] for result in data["results"]) for data in comp_data),
        "output_tokens": sum(sum(result["output_tokens"] for result in data["results"]) for data in comp_data)
    }

    # URL of OpenAI Admin Completions Usage API endpoint.
    emb_url = "https://api.openai.com/v1/organization/usage/embeddings"

    emb_data = get_openai_usage(openai_admin_key, start_time, end_time, emb_url)
    embeddings = {"input_tokens": sum(sum(result["input_tokens"] for result in data["results"]) for data in emb_data)}

    # URL of OpenAI Admin Vector Stores Usage API endpoint.
    vec_url = "https://api.openai.com/v1/organization/usage/vector_stores"

    vec_data = get_openai_usage(openai_admin_key, start_time, end_time, vec_url)
    vector_store = {"usage_bytes": sum(sum(result["usage_bytes"] for result in data["results"]) for data in vec_data)}

    return {
        "completions": completions,
        "embeddings": embeddings,
        "vector_store": vector_store
    }


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

# Load the conversation from the file.
with open(args.conversation, mode='r', encoding=ENC) as conversation_file:
    conversation = json.load(conversation_file)

# Select a subset of the conversation if desired.
if args.test_cases is not None:
    conversation = select(conversation, args.test_cases)

# Have a conversation with the chatbot.
responses = [chatbot.invoke(exchange["query"]) for exchange in conversation]

# Print time information.
end_time = datetime.now()
run_time = end_time - start_time
logger.info(f"Time: {run_time}")
logger.info(f"Average time per query: {run_time / len(conversation)}")

# Print cost information.
cost = get_cost(os.getenv("OPENAI_ADMIN_KEY"), start_time, end_time)
logger.info(f"Cost: ${cost}")
logger.info(f"Average cost per query: ${cost / len(conversation)}")

# Print token usage information.
token_usage = get_token_usage(os.getenv("OPENAI_ADMIN_KEY"), start_time, end_time)
logger.info(f"Completions token usage:\n{token_usage}")
avg_token_usg = {c: {k: int(v / len(conversation)) for k, v in cv.items()} for c, cv in token_usage.items()}
logger.info(f"Average completions token usage per query:\n{avg_token_usg}")

# Print transcript of conversation.
logger.info("Chat Transcript:\n{}\n".format(
    "\n".join([
        "USR> {}\nBOT> {}".format(conversation[i]["query"], responses[i])
        for i in range(len(conversation))
        ])
    ))

# Print evaluation results.
for e in evaluators:
    logger.info(e.score_conversation(responses, conversation))

print(f"Ran test in {datetime.now() - start_time}")
print(f"Output in {args.output_log}")
