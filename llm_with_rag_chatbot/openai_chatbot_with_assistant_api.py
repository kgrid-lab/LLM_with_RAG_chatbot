import json
import os
import re
from collections import deque

import openai
import requests
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL")
knowledge_base = os.getenv("KNOWLEDGE_BASE")

# Setup OpenAI API client
openai.api_key = OPENAI_API_KEY

# Initialize the language model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model_name)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
splits = []
files = os.listdir(knowledge_base)
for file_name in files:
    if file_name == 'code':
        continue
    loader = TextLoader(os.path.join(knowledge_base, file_name))
    ko = loader.load()

    with open(os.path.join(knowledge_base, file_name), "r") as file:
        code_file = json.load(file)
    link = code_file["koio:hasKnowledge"]["implementedBy"]
    response = requests.get(link)
    data = response.text
    ko[0].page_content += (
        "Here is the function to calculate the value for this knowledge object: \n "
        + data
    )
    splits.extend(ko)
vectorstore2 = DocArrayInMemorySearch.from_documents(splits, embeddings)

# Create the Chain
template = """
Use the context to prepare the code to be executed to answer the question. \
If not provided, ask the user for the input values that are needed to execute the code.
Do not return code unless you have received the input values.
Include the input values provided by the user in the code.
Include the sentences that could be used with the result of the code execution to answer the question.

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()
chain = (
    {"context": vectorstore2.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# OpenAI Assistants API setup
assistant = openai.beta.assistants.create(
    name="Code Executor",
    instructions="You are a code executor. Execute the provided code and return the response.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview",
)


# Function to execute code using Assistants API
def execute_code_with_assistant(code):
    client = openai.OpenAI()
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=code,
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please execute the provided code and use the result value together with the narrative part to answer the question. Do not mention that you executed code to provide the response.",
    )

    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        result = ""
        for message in messages:
            if message.role == "assistant" and message.content[0].type == "text":
                result = message.content[0].text.value
        return result
    else:
        return "Code execution failed."


# Prepare History
def get_full_context(history, current_query):
    history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
    full_context = f"{history_text}\nUser: {current_query}\nBot:"
    return full_context


def process(text, conversation_history):
    full_context = get_full_context(conversation_history, text)
    response = chain.invoke(full_context)
    code = (
        re.search(r"```(.*?)```", response, re.DOTALL).group(1)
        if "```" in response
        else ""
    )
    if code:
        print("I am processing your request, this may take a few seconds...")
        execution_result = execute_code_with_assistant(response)
        return execution_result
    else:
        return response


def main():
    # Store the conversation history
    conversation_history = deque(maxlen=10)

    while True:
        text = input("Enter your query: --> ")
        response = process(text, conversation_history)
        print(response)
        code = (
            re.search(r"```(.*?)```", response, re.DOTALL).group(1)
            if "```" in response
            else ""
        )
        conversation_history.append(
            (text, response.replace(code, ""))
        )  # update history excluding code


if __name__ == "__main__":
    main()

# Example request: Can you calculate my life year gain if I stop using tobacco considering I am a 65 years old female that has been smoking for 10 years now and I still smoke and I smoke 5 cigarettes a day
