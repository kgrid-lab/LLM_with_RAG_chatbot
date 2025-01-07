import os

from dotenv import load_dotenv

from chatbots import LlmWithRagKosAndExternalInterpreter

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL")
knowledge_base = os.getenv("KNOWLEDGE_BASE")
model_seed = int(os.getenv("MODEL_SEED"))

def main():
    chatbot = LlmWithRagKosAndExternalInterpreter(OPENAI_API_KEY, model_name, model_seed, knowledge_base)

    while True:
        text = input("Enter your query: --> ")
        response = chatbot.invoke(text)
        print(response)

if __name__ == "__main__":
    main()

# Example request: Can you calculate my life year gain if I stop using tobacco considering I am a 65 years old female that has been smoking for 10 years now and I still smoke and I smoke 5 cigarettes a day
