import logging

import openai

from . import Chatbot


PROMPT = """
You are an assistant helping the clinician user perform clinicial calculations.
Follow these steps:
1. Read the user's question to find out if they are requesting you to perform a calculation.
    - If the user is requesting a calculation, identify which of the following functions can be used to perform this calculation. If there are multiple functions which can perform this calculation, stop and ask the user which function to use before proceeding.
        - Functions:
            1. ASCVD (Atherosclerotic Cardiovascular Disease) 2013 Risk Calculator
            2. Body Mass Index (BMI)
            3. Body Surface Area (BSA)
            4. CHA2DS2-VASc Score
            5. 2021 CKD-EPI equation for Glomerular Filtration Rate (GFR)
            6. Cockcroft-Gault Creatinine Clearance
            7. Corrected Calcium for Hypo- or Hyperalbuminemia
            8. MDRD GFR Equation
            9. Mean Arterial Pressure
            10. NIH Stroke Scale
            11. Wells' Criteria for Pulmonary Embolism
    - If the user is not requesting a calculation, simply answer their query and do not proceed with the remaining steps.
2. Identify all the parameters of this function.
3. Of these parameters, identify which parameters have not been assigned a value.
4. For each parameter which has not yet been assigned a value:
    - If the parameter is optional, tell the user this parameter is optional and ask if they would like to provide a value. If they decline, assign this parameter a value of null.
    - If the parameter is required, ask the user to provide a value for this parameter. Once the user provides a value, assign this parameter to the value the user provided.
5. Once all parameters have been assigned values, check the units of these values. If the user provided parameter values in different units than what the function requires, convert these values to the units required by the function.
6. Call the function with the final set of parameter values and return the result to the user.
"""

logger = logging.getLogger(__name__)


class PlainLlm(Chatbot):
    """
    This chatbot uses only a zero-shot LLM, without RAG or tools.
    The only information about the task at hand is in the prompt.
    """

    def __init__(self, openai_api_key: str, model_name: str):
        """
        Constructor
        Parameters:
        - openai_api_key: See README on how to get one. Recommend putting it in .env.
        - model_name: Which model to use. Only OpenAI is currently supported. Recommend storing in .env.
        """

        # Initialize OpenAI client.
        self._client = openai.OpenAI(api_key=openai_api_key)

        # Save model name.
        self._model_name = model_name

        # Build a list of messages for the LLM to respond to.
        # The list will begin with a system message containing the prompt.
        self._messages = [{
            "content": PROMPT,
            "role": "system"
        }]

    def invoke(self, query: str) -> str:
        # Add the user's query to the end of the message list.
        self._messages.append({
            "content": query,
            "role": "user"
        })

        # Ask the LLM to respond to the messages using the OpenAI Chat Completions API.
        response = self._client.chat.completions.create(
            messages=self._messages,
            model=self._model_name,
            temperature=0
        )

        # Process the response.
        if len(response.choices) != 1:
            logger.error("Unexpected number of choice entries in Chat Completions response {}".format(response))
            return "ERROR"
        response_entry = response.choices[0]

        # If the LLM is done, return the response.
        if response_entry.finish_reason == "stop":
            self._messages.append(response_entry.message)
            return response_entry.message.content
        else:
            logger.error("Unexpected status {} in Chat Completions response after tool call {}".format(response_entry.finish_reason, response_entry))
            return "ERROR"