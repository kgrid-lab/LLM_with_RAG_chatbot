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
import csv
from datetime import datetime, timedelta
from itertools import chain
import json
import logging
from math import ceil
import os
import requests
from statistics import mean, stdev
import time

from dotenv import load_dotenv

from evaluators import KeywordEvaluator, BleuEvaluator, Rouge1Evaluator, Rouge2Evaluator, RougeLEvaluator, LlmEvaluator
from src.chatbots import init_chatbot_from_str, chatbot_options

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

def get_cost(token_usage: dict, model: str, emb_model: str) -> float:
    """Gets the OpenAI API usage cost in US dollars given token usage."""

    M = 1000000
    G = 1000000000
    COST = {
        "gpt-4o": {
            "input_tokens": 2.50 / M,
            "input_cached_tokens": 1.25 / M,
            "output_tokens": 10.00 / M
        },
        "text-embedding-3-small": {
            "input_tokens": 0.020 / M
        },
        "text-embedding-ada-002": {
            "input_tokens": 0.100 / M
        },
        "file_search": {
            "usage_bytes": 0.10 / G
        }
    }

    uncached_input_tokens = token_usage["completions"]["input_tokens"] - token_usage["completions"]["input_cached_tokens"]

    return sum([
        uncached_input_tokens * COST[model]["input_tokens"],
        token_usage["completions"]["input_cached_tokens"] * COST[model]["input_cached_tokens"],
        token_usage["completions"]["output_tokens"] * COST[model]["output_tokens"],
        token_usage["embeddings"]["input_tokens"] * COST[emb_model]["input_tokens"],
        max(0, (token_usage["vector_store"]["usage_bytes"] - G) * COST["file_search"]["usage_bytes"])
    ])


def get_token_usage(openai_admin_key: str, start_time: datetime, end_time: datetime) -> dict:
    """
    Returns an object containing relevant token and byte usage statistics.
    Note that this function is quite slow because two minutes are spent sleeping.
    """

    # Necessary to wait 1 minute because minimum granularity of OpenAI
    # token usage reporting is 1 minute.
    # We wait to ensure that no usage information from the previous run
    # spills into reporting for the current run.
    time.sleep(60)

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

    # We wait 1 minutes again here because the minimum granularity of OpenAI
    # token usage reporting is 1 minute.
    # If there is less spacing after getting the usage and the start of the
    # next run, usage information from this run can spill into reporting
    # of the next run.
    time.sleep(60)

    return {
        "completions": completions,
        "embeddings": embeddings,
        "vector_store": vector_store
    }

def run_test(conversation: str,
             chatbot_architecture: str,
             openai_api_key: str,
             model_name: str,
             model_seed: str,
             knowledge_base: str,
             embedding_model: str,
             embedding_dimension: int) -> dict:
    """
    Runs the specified chatbot with the specific conversation.
    Returns a dict containing key summary results.
    Item-by-item results and other info are printed to log.
    """
    # Log start time.
    start_time = datetime.now()

    # Initialize chatbot.
    chatbot = init_chatbot_from_str(
        chatbot_architecture,
        openai_api_key,
        model_name,
        model_seed,
        knowledge_base,
        embedding_model,
        embedding_dimension
    )

    # Have a conversation with the chatbot.
    responses = [chatbot.invoke(exchange["query"]) for exchange in conversation]

    # Record end time
    end_time = datetime.now()

    # Build result object.
    result = {}

    # Log model and architecture information.
    logger.info("Model and Architecture Information:\n")
    logger.info("Architecture: {})\n".format(chatbot.get_architecture()))
    logger.info("Model name: {}\n".format(model_name))
    logger.info("Model seed: {}\n".format(model_seed))
    logger.info("RAG Knowledge Base: {}\n".format(knowledge_base))
    logger.info("\n")

    # Log time information.
    run_time = end_time - start_time
    logger.info(f"Time: {run_time}")
    result["Time"] = run_time
    avg_time = run_time / len(conversation)
    logger.info(f"Average time per query: {avg_time}")
    result["Time per query"] = avg_time

    # Log token usage information.
    token_usage = get_token_usage(os.getenv("OPENAI_ADMIN_KEY"), start_time, end_time)
    logger.info(f"Token usage:\n{token_usage}")
    avg_token_usg = {c: {k: int(v / len(conversation)) for k, v in cv.items()} for c, cv in token_usage.items()}
    logger.info(f"Average completions token usage per query:\n{avg_token_usg}")
    result["Input tokens"] = token_usage["completions"]["input_tokens"]
    result["Input tokens per query"] = avg_token_usg["completions"]["input_tokens"]
    result["Cached input tokens"] = token_usage["completions"]["input_cached_tokens"]
    result["Cached input tokens per query"] = avg_token_usg["completions"]["input_cached_tokens"]
    result["Output tokens"] = token_usage["completions"]["output_tokens"]
    result["Output tokens per query"] = avg_token_usg["completions"]["output_tokens"]

    # Log cost information.
    cost = get_cost(token_usage, model_name, embedding_model)
    logger.info(f"Cost: ${cost}")
    result["Cost"] = cost
    avg_cost = cost / len(conversation)
    logger.info(f"Average cost per query: ${avg_cost}")
    result["Cost per query"] = avg_cost

    # Log transcript of conversation.
    logger.info("Chat Transcript:\n{}\n".format(
        "\n".join([
            "USR> {}\nBOT> {}".format(conversation[i]["query"], responses[i])
            for i in range(len(conversation))
            ])
        ))

    # Log evaluation results.
    evaluators = (
        KeywordEvaluator(),
        BleuEvaluator(),
        Rouge1Evaluator(),
        Rouge2Evaluator(),
        RougeLEvaluator(),
        LlmEvaluator(openai_api_key, os.getenv("EVAL_MODEL"))
    )
    for e in evaluators:
        e_result = e.score_conversation(responses, conversation)
        result[e.get_name()] = e_result["key_result"]
        logger.info(e_result["verbose"])

    return result

def compute_mean_and_standard_deviation(vals) -> tuple:
    if isinstance(vals[0], timedelta):
        m = sum(vals, start=timedelta()) / len(vals)
        vals_sec = [v.total_seconds() for v in vals]
        sd_sec = stdev(vals_sec)
        sd = timedelta(seconds=sd_sec)
    else:
        m = mean(vals)
        sd = stdev(vals)
    return (m, sd)

# Log script start time for reporting purposes.
script_start_time = datetime.now()
timestamp = script_start_time.strftime(TIME_FMT)

# Print message so that user does not think program has frozen.
print("Chatbot test running...\n")

# Load environment variables.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL")
knowledge_base = os.getenv("KNOWLEDGE_BASE")
model_seed = int(os.getenv("MODEL_SEED"))
embedding_model = os.getenv("EMBEDDING_MODEL")
embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION"))

# Process command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--chatbot_architectures", "-a", nargs="+", type=str, choices=chatbot_options, required=True, help="Which chatbot architecture to test.")
parser.add_argument("--conversations", "-c", nargs="+", type=str, required=True, help="Text files containing prepared conversations.")
parser.add_argument("--num_trials", "-n", default=1, type=int, help="Number of times to run each test. Default is 1.")
parser.add_argument("--test_cases", "-t", nargs="+", type=int, help="If desired, specify which test cases to run. The first is #0. Default is to run all tests.")
parser.add_argument("--output_log", "-o",
                    default=f"chatbot_test_output_{timestamp}.log",
                    type=str, help="If desired, specify a file other than the default to which to log the chatbot test output.")
parser.add_argument("--output_csv", "-v",
                    default=f"chatbot_test_output_{timestamp}.csv",
                    type=str, help="If desired, specify a file other than the default to which log key results in CSV format.")
parser.add_argument("--log_level", "-l", default="INFO", type=str, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                    help="If desired, specify a logging level other than the default of INFO.")
args = parser.parse_args()

# Set up logging.
logger = logging.getLogger(__name__)
logging.basicConfig(filename=args.output_log, filemode="w", level=logging.getLevelNamesMapping()[args.log_level], encoding=ENC)

# Run the test conversation for each conversation and architecture and save key results.
key_results = {}
for convo_path in args.conversations:
    key_results[convo_path] = {}

    # Load conversation from file
    with open(convo_path, mode='r', encoding=ENC) as convo_file:
        convo_data = json.load(convo_file)

        # Select a subset of the conversation if desired.
        if args.test_cases is not None:
            convo_data = select(convo_data, args.test_cases)

    # For this conversation, run the chatbot on each architecture.
    for chatbot_architecture in args.chatbot_architectures:

        # Run this chatbot architecture on this conversation the specified number of times.
        results = []
        for trial in range(args.num_trials):
            result = run_test(
                convo_data,
                chatbot_architecture,
                OPENAI_API_KEY,
                model_name,
                model_seed,
                knowledge_base,
                embedding_model,
                embedding_dimension
            )

            # Log results of this run.
            logger.info(f"Result of trial {trial} of running architecture {chatbot_architecture} on conversation {convo_path}:\n{result}")
            results.append(result)

        # Aggregate the results of multiple runs.
        key_result = {}
        for field in results[0].keys():
            vals = [r[field] for r in results]
            m, sd = compute_mean_and_standard_deviation(vals)
            key_result[field] = f"{m} +/- {sd}"

        # Log aggregated result.
        logger.info(f"Aggregated results of running architecture {chatbot_architecture} on conversation {convo_path}:\n{key_result}")
        key_results[convo_path][chatbot_architecture] = key_result

with open(args.output_csv, "w", encoding=ENC, newline='') as ocsv:
    csv_writer = csv.writer(ocsv)
    csv_writer.writerow(["Conversation"] + list(chain.from_iterable([([convo] + [""] * (len(args.chatbot_architectures) - 1)) for convo in args.conversations])))
    csv_writer.writerow([""] + args.chatbot_architectures * len(args.conversations))
    for row_name in key_results[args.conversations[0]][args.chatbot_architectures[0]].keys():
        csv_writer.writerow([row_name] + list(chain.from_iterable([[key_results[convo][arch][row_name] for arch in args.chatbot_architectures] for convo in args.conversations])))

print(f"Ran test in {datetime.now() - script_start_time}")
print(f"Data in {args.output_csv}")
print(f"Logs in {args.output_log}")
