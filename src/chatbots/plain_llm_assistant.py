import logging

import openai

from . import Chatbot


logger = logging.getLogger(__name__)


def get_latest_response(client, thread_id, run_id) -> str:
    """
    Utility function to get the most recent response from an OpenAI assistant.
    """
    messages = client.beta.threads.messages.list(
        thread_id=thread_id, limit=1, order="desc", run_id=run_id
    )
    return messages.data[0].content[0].text.value


class PlainLlmAssistant(Chatbot):
    """Like PlainLlm except using the OpenAI Assistants API."""

    def __init__(self, openai_api_key: str, model_name: str):
        """
        Constructor
        Parameters:
        - openai_api_key: See README on how to get one. Recommend putting it in .env.
        - model_name: Which model to use. Only OpenAI is currently supported. Recommend storing in .env.
        """

        # Initialize OpenAI client.
        self._client = openai.OpenAI(api_key=openai_api_key)

        # Initialize an OpenAI assistant.
        self._assistant = self._client.beta.assistants.create(
            name="Clinical Calculator",
            instructions="""
You are an assistant helping the clinician user perform clinicial calculations.
Follow these steps:
Step 1: Read the user's question and identify which calculation they are requesting you to perform, if any.
        If no calculation is being requested, simply answer their question. You are done.
        If there are multiple methods available to perform the requested calculation, stop and ask the user which one they would like to use.
Step 2: Identify all the parameters required by the calculation.
Step 3: Identify which of these parameters the user has not provided values for.
Step 4: For each of these parameters, ask the user to provide the value.
Step 5: Once values have been obtained for all parameters, perform the calculation.
Step 6: Communicate the result of the calculation to the user.
            """,
            model=model_name,
            temperature=0,
        )

        # Create a thread, which represents a conversation between
        # the clinician and the assistant.
        self._thread = self._client.beta.threads.create()

    def invoke(self, query: str, session_id: str) -> str:
        # Add a message to the thread containing the clinician's query.
        self._client.beta.threads.messages.create(
            thread_id=self._thread.id, role="user", content=query
        )

        # Asking the LLM to respond to the query is an asynchronous task,
        # so we simply wait until it is done (i.e. create_and_poll).
        run = self._client.beta.threads.runs.create_and_poll(
            thread_id=self._thread.id, assistant_id=self._assistant.id
        )

        # If the LLM finishes without needing to call a tool, return the response.
        # If the LLM requires a tool call, call the python function and feed the LLM
        # the result.
        if run.status == "completed":
            return get_latest_response(self._client, self._thread.id, run.id)
        else:
            logger.error(
                "Run failed on query {} with status {}\n".format(query, run.status)
            )
            return ""