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
from collections.abc import Mapping, Sequence
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
from typing import Any, Tuple

from dotenv import load_dotenv

from evaluators import EvaluationResult, KeywordEvaluator, BleuEvaluator, Rouge1Evaluator, Rouge2Evaluator, RougeLEvaluator, LlmEvaluator
from src.chatbots import init_chatbot_from_str, chatbot_options

TIME_FMT = "%Y-%m-%d-%H%M.%S.%f"
ENC = "utf-8"


class ChatbotPerformance:
    """Holds data about the performance of a given chatbot on a given conversation."""

    """List of performance metrics. Class attribute."""
    METRICS = [
        "Time",
        "Time per query",
        "Input tokens",
        "Input tokens per query",
        "Cached input tokens",
        "Cached input tokens per query",
        "Output tokens",
        "Output tokens per query",
        "Cost",
        "Cost per query"
    ]

    def __init__(self, num_queries: int, time: timedelta, input_tokens: int, cached_input_tokens: int, output_tokens: int, cost: float):
        """
        Stores the given chatbot performance metrics in a ChatbotPerformance object.
        Also computes per query and cost metrics.
        """
        self._dict = {
            self.METRICS[0]: time,
            self.METRICS[1]: time / num_queries,
            self.METRICS[2]: input_tokens,
            self.METRICS[3]: input_tokens / num_queries,
            self.METRICS[4]: cached_input_tokens,
            self.METRICS[5]: cached_input_tokens / num_queries,
            self.METRICS[6]: output_tokens,
            self.METRICS[7]: output_tokens / num_queries,
            self.METRICS[8]: cost,
            self.METRICS[9]: cost / num_queries,
        }

    def as_dict(self) -> Mapping[str, Any]:
        """Return the data contined by this object in dictionary format."""
        return self._dict


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

def run_test(conversation: Sequence[Mapping[str, Any]],
             chatbot_architecture: str,
             openai_api_key: str,
             model_name: str,
             model_seed: int,
             knowledge_base: str,
             embedding_model: str,
             embedding_dimension: int) -> Tuple[Sequence[str], ChatbotPerformance, Sequence[EvaluationResult]]:
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

    # Log model and architecture information.
    logger.info("Model and Architecture Information:\n")
    logger.info("Architecture: {})\n".format(chatbot.get_architecture()))
    logger.info("Model name: {}\n".format(model_name))
    logger.info("Model seed: {}\n".format(model_seed))
    logger.info("RAG Knowledge Base: {}\n".format(knowledge_base))
    logger.info("\n")

    # Collect performance information including time, token usage, and cost.
    run_time = end_time - start_time
    token_usage = get_token_usage(os.getenv("OPENAI_ADMIN_KEY"), start_time, end_time)
    perf_data = ChatbotPerformance(
        num_queries=len(conversation),
        time=run_time,
        input_tokens=token_usage["completions"]["input_tokens"],
        cached_input_tokens=token_usage["completions"]["input_cached_tokens"],
        output_tokens=token_usage["completions"]["output_tokens"],
        cost=get_cost(token_usage, model_name, embedding_model)
    )

    # Log transcript of conversation.
    logger.info("Chat Transcript:\n{}\n".format(
        "\n".join([
            "USR> {}\nBOT> {}".format(conversation[i]["query"], responses[i])
            for i in range(len(conversation))
            ])
        ))

    # Return evaluation results.
    evaluators = (
        KeywordEvaluator(),
        BleuEvaluator(),
        Rouge1Evaluator(),
        Rouge2Evaluator(),
        RougeLEvaluator(),
        LlmEvaluator(openai_api_key, os.getenv("EVAL_MODEL"))
    )

    return responses, perf_data, [e.score_conversation(responses, conversation) for e in evaluators]

def compute_mean_and_standard_deviation(vals: Sequence[timedelta | float]) -> Tuple[timedelta | float, timedelta | float]:
    if isinstance(vals[0], timedelta):
        if len(vals) == 1:
            m = vals[0]
            sd = timedelta()
        else:
            m = sum(vals, start=timedelta()) / len(vals)
            vals_sec = [v.total_seconds() for v in vals]
            sd_sec = stdev(vals_sec)
            sd = timedelta(seconds=sd_sec)
    else:
        if len(vals) == 1:
            m = vals[0]
            sd = 0
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

# Run the test conversation for each conversation and architecture and save results.
key_results = {}
key_result_rows = ChatbotPerformance.METRICS
key_result_rows.extend((
        KeywordEvaluator().get_name(),
        BleuEvaluator().get_name(),
        Rouge1Evaluator().get_name(),
        Rouge2Evaluator().get_name(),
        RougeLEvaluator().get_name(),
        LlmEvaluator(OPENAI_API_KEY, os.getenv("EVAL_MODEL")).get_name()
))
cat_ids = set()
item_results: Sequence[Mapping] = []
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
        key_result_components = []
        for trial in range(args.num_trials):
            responses, perf_results, eval_results = run_test(
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
            logger.info(f"Result of trial {trial} of running architecture {chatbot_architecture} on conversation {convo_path}:\n{eval_results}")
            key_result_component = perf_results.as_dict()
            logger.info(f"Performance results:\n{perf_results}")
            item_result_batch = [
                {
                    "conversation": convo_path,
                    "architecture": chatbot_architecture,
                    "trial": trial,
                    "item": i,
                    "query": convo_data[i]["query"],
                    "response": responses[i]
                }
                for i in range(len(responses))
            ]
            for eval_result in eval_results:
                key_result_component[eval_result.evaluator_name] = eval_result.overall_result
                for cat_name, cat_val in eval_result.category_results.items():
                    cat_id = f"{eval_result.evaluator_name}_{cat_name}"
                    cat_ids.add(cat_id)
                    key_result_component[cat_id] = cat_val
                for item_num, score in enumerate(eval_result.item_results):
                    item_result_batch[item_num][eval_result.evaluator_name] = score
            key_result_components.append(key_result_component)
            item_results.extend(item_result_batch)

        # Aggregate the results of multiple runs.
        key_result = {}
        for field in key_result_components[0].keys():
            vals = [r[field] for r in key_result_components]
            m, sd = compute_mean_and_standard_deviation(vals)
            key_result[field] = f"{m} +/- {sd}"

        # Log aggregated result.
        logger.info(f"Aggregated results of running architecture {chatbot_architecture} on conversation {convo_path}:\n{key_result}")
        key_results[convo_path][chatbot_architecture] = key_result

key_result_rows.extend(cat_ids)

with open(args.output_csv, "w", encoding=ENC, newline='') as ocsv:
    csv_writer = csv.writer(ocsv)

    # Write key results.
    csv_writer.writerow(["Conversation"] + list(chain.from_iterable([([convo] + [""] * (len(args.chatbot_architectures) - 1)) for convo in args.conversations])))
    csv_writer.writerow([""] + args.chatbot_architectures * len(args.conversations))
    for row_name in key_result_rows:
        csv_writer.writerow([row_name] + list(chain.from_iterable([[key_results[convo][arch].get(row_name, "") for arch in args.chatbot_architectures] for convo in args.conversations])))

    # Write item-by-item results. (different number of columns than key results, and could be thousands of rows)
    header = item_results[0].keys()
    csv_writer.writerow(header)
    for item_result in item_results:
        csv_writer.writerow([item_result[col] for col in header])

print(f"Ran test in {datetime.now() - script_start_time}")
print(f"Data in {args.output_csv}")
print(f"Logs in {args.output_log}")
