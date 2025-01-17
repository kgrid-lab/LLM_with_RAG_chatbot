import json
import logging
import os
import re
from collections import deque

import openai
import requests
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from . import Chatbot

ENC = "utf-8"

logger = logging.getLogger(__name__)


def get_full_context(history, current_query):
    """
    Utility function to prepare a string containing the context
    (conversation history) for the LLM.
    """
    history_text = "\n".join([f"Clinician: {q}\nYou: {a}" for q, a in history])
    full_context = f"{history_text}\nClinician: {current_query}\nYou:"
    return full_context


def extract_code(response: str) -> str:
    """
    Utility function to find and extract code from chatbot response.
    If no code is found, returns the empty string.
    """
    if "```" in response:
        search_result = re.search(r"```(.*?)```", response, re.DOTALL)
        if search_result is not None:
            return search_result.group(1)
        else:
            return ""
    else:
        return ""


class LlmWithRagKosAndInternalExecution(Chatbot):
    """
    Through RAG, the main LLM of this Chatbot has access to a knowledge_base of
    Knowledge Object (KO) JSON metadata files. These KO metadata files describe
    the KOs and contain a hyperlink to a Python code file describing the precise
    calculation or formula referred to by the KO. When asked to perform a
    calculation, this Chatbot instructs the main LLM to fetch and return the
    Python code for that calculation. This chatbot then executes the code 
    internally and engages the LLM again to prepare the final response.
    """

    def __init__(
        self, openai_api_key: str, model_name: str, model_seed: int, knowledge_base: str
    ):
        """
        Constructor
        Parameters:
        - openai_api_key: See README on how to get one. Recommend putting it in .env.
        - model_name: Which model to use. Only OpenAI is currently supported. Recommend storing in .env.
        - model_seed: Specify seed for reproducibility. Recommend storing in .env.
        - knowledge_base: Directory containing Knowledge Object (KO) JSON metadata files. Recommend storing in .env.
        """

        # Setup OpenAI API client
        openai.api_key = openai_api_key

        # Initialize the language model
        model = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=model_name,
            temperature=0,
            seed=model_seed,
        )

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        splits = []
        file_paths = (
            file.path for file in os.scandir(knowledge_base) if file.is_file()
        )
        for file_path in file_paths:
            loader = TextLoader(file_path, encoding=ENC)
            ko = loader.load()

            with open(file_path, "r", encoding=ENC) as file:
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
        You are an assistant helping a clinician perform calculations.
        However, you do not perform the calculations yourself.
        Instead, follow the steps below:
        Step 1: Read the clinician's question (labeled "Question:" below) and identify the calculation the clinician is requesting, if any.
        Step 2: Look at the information below (labeled "Information:") to find:
                - The Python code implementing the logic of the calculation
                - The parameters required by that code.
        Step 3: Gather values for all the required parameters.
                - If the clinician has not provided values for all the required parameters, please ask them for the missing values.
                - Some parameters are optional. If the clinician does not provide values for these optional parameters, please notify them that they are optional and confirm whether they want to proceed without them.
                - Sometimes, the clinician might provide values in different units than what the code requires. In this case, please convert them to the units required by the code.
        Step 4: Once you have values for all the required parameters, provide the code to execute, wrapping the function in ```function code ``` and the function call with input values (without assigning it to a variable) to execute it in ```line code ```

        Question: {question}

        Information: {info}
        """
        prompt = ChatPromptTemplate.from_template(template)
        parser = StrOutputParser()
        self._chain = (
            {
                "info": vectorstore2.as_retriever(search_kwargs={"k": 20}),
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
            | parser
        )

        # Store the conversation history
        self._conversation_history = deque(maxlen=10)

    def process(self, text, conversation_history):
        logger.info(
            "Received input:\n%s\nWith history:\n%s\n",
            text,
            "\n".join(
                ("USR> {}\nBOT> {}".format(h[0], h[1]) for h in conversation_history)
            ),
        )
        full_context = get_full_context(conversation_history, text)
        response = self._chain.invoke(full_context)

        function = (
            re.search(r"```function code\n(.*?)```", response, re.DOTALL).group(1)
            if "```" in response
            else ""
        )
        if function:
            call = re.search(r"```line code\n(.*?)```", response, re.DOTALL).group(1)
            print("I am processing your request, this may take a few seconds...")
            exec(function)
            result = eval(call)
            conversation_history.append(
                (text, response.replace(function, ""))
            )
            text = "I ran the code and the result is " +  str(result)+". pretend you ran the code and use this value to provide a short final response to the last question that required code execution. Do not include any code."
            full_context = get_full_context(conversation_history, text)
            return self._chain.invoke(full_context)
        else:
            return response

    def get_architecture(self) -> str:
        return "LLM with RAG KOs and external evaluator"

    def invoke(self, query: str) -> str:
        response = self.process(query, self._conversation_history)

        code = extract_code(response)
        self._conversation_history.append(
            (query, response.replace(code, ""))
        )  # update history excluding code

        return response
