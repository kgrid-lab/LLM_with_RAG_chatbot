"""
Command-line interface for chatbot.

Can specify which chatbot to use when launching with -a.
"""

import argparse
import os

from dotenv import load_dotenv

from chatbots import init_chatbot_from_str, chatbot_options

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL")
knowledge_base = os.getenv("KNOWLEDGE_BASE")
model_seed = int(os.getenv("MODEL_SEED"))
embedding_model = os.getenv("EMBEDDING_MODEL")
embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION"))

parser = argparse.ArgumentParser(description="Command-line interface for chatbot.")
parser.add_argument(
    "--chatbot_architecture",
    "-a",
    default="LlmWithKoCodeTools",
    type=str,
    choices=chatbot_options,
    help="Which chatbot architecture to use.",
)
args = parser.parse_args()


def main():
    chatbot = init_chatbot_from_str(args.chatbot_architecture, OPENAI_API_KEY, model_name, model_seed, knowledge_base, embedding_model, embedding_dimension)

    while True:
        text = input("Enter your query: --> ")
        response = chatbot.invoke(text)
        print(response)


if __name__ == "__main__":
    main()

# Example request: Can you calculate my life year gain if I stop using tobacco considering I am a 65 years old female that has been smoking for 10 years now and I still smoke and I smoke 5 cigarettes a day
