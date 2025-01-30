import json
import logging
import os
import re
from collections import deque

import openai
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

import KO.clinical_calculators.code.nihss as nihss

from KO.clinical_calculators.code.ascvd_2013 import ascvd_2013
from KO.clinical_calculators.code.bmi import bmi
from KO.clinical_calculators.code.bsa import bsa
# from KO.clinical_calculators.code.chadsvasc import chadsvasc
# from KO.clinical_calculators.code.ckd_epi_gfr_2021 import ckd_epi_gfr_2021
# from KO.clinical_calculators.code.cockcroft_gault_cr_cl import cockcroft_gault_cr_cl
# from KO.clinical_calculators.code.corr_ca_alb import corr_ca_alb
# from KO.clinical_calculators.code.mdrd_gfr import mdrd_gfr
# from KO.clinical_calculators.code.mean_arterial_pressure import mean_arterial_pressure
# from KO.clinical_calculators.code.wells import wells
from . import Chatbot



def nihss_adapter(
    consciousness: str,
    month_and_age_questions: str,
    blink_eyes_and_squeeze_hands: str,
    horizontal_extraocular_movements: str,
    visual_fields: str,
    facial_palsy: str,
    left_arm_motor_drift: str,
    right_arm_motor_drift: str,
    left_leg_motor_drift: str,
    right_leg_motor_drift: str,
    limb_ataxia: str,
    sensation: str,
    language: str,
    dysarthria: str,
    inattention: str,
) -> int:
    return nihss.nihss(
        nihss.Consciousness[consciousness],
        nihss.MonthAndAgeQuestions[month_and_age_questions],
        nihss.BlinkEyesAndSqueezeHands[blink_eyes_and_squeeze_hands],
        nihss.HorizontalExtraocularMovements[horizontal_extraocular_movements],
        nihss.VisualFields[visual_fields],
        nihss.FacialPalsy[facial_palsy],
        nihss.ArmMotorDrift[left_arm_motor_drift],
        nihss.ArmMotorDrift[right_arm_motor_drift],
        nihss.LegMotorDrift[left_leg_motor_drift],
        nihss.LegMotorDrift[right_leg_motor_drift],
        nihss.LimbAtaxia[limb_ataxia],
        nihss.Sensation[sensation],
        nihss.LanguageAphasia[language],
        nihss.Dysarthria[dysarthria],
        nihss.ExtinctionInattention[inattention],
    )


CODE_MAP = {
    "ascvd-2013": ascvd_2013,
    "bmi": bmi,
    "bsa": bsa,
    # "chadsvasc": chadsvasc,
    # "ckd-epi-gfr-2021": ckd_epi_gfr_2021,
    # "cockcroft-gault-cr-cl": cockcroft_gault_cr_cl,
    # "corr-ca-alb": corr_ca_alb,
    # "mdrd-gfr": mdrd_gfr,
    # "mean-arterial-pressure": mean_arterial_pressure,
    # "nihss": nihss_adapter,
    # "wells": wells,
}

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


class LlmWithRagMetadataWithEmbeddedCodeChatbot(Chatbot):
    """
    Through RAG, the main LLM of this Chatbot has access to a knowledge_base of
    Knowledge Object (KO) JSON metadata files. These KO metadata files describe
    the KOs and contain a hyperlink to a Python code file describing the precise
    calculation or formula referred to by the KO. When asked to perform a
    calculation, this Chatbot instructs the main LLM to fetch and return the
    input parameters and function name required for that calculation. This chatbot then executes the code 
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

            splits.extend(ko)
        vectorstore2 = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Create the Chain
        template = """
        You are an assistant helping a clinician perform calculations.
        However, you do not perform the calculations yourself.
        Instead, follow the steps below:
        Step 1: Read the clinician's question (labeled "Question:" below) and identify the calculation the clinician is requesting, if any.
        Step 2: Look at the information below (labeled "Information:") to find:
                - The Python function implementing the logic of the calculation
                - The parameters required by that function.
        Step 3: Gather values for all the required parameters.
                - If the clinician has not provided values for all the required parameters, please ask them for the missing values.
                - Some parameters are optional. If the clinician does not provide values for these optional parameters, please notify them that they are optional and confirm whether they want to proceed without them.
                - Sometimes, the clinician might provide values in different units than what the function requires. In this case, please convert them to the units required by the function.
        Step 4: Once you have values for all the required parameters, provide the input parameters in json format, wrapping it in ```input=\n[input value]\n``` and the function name in ```function=\n[function name]\n```. Replace the text in bracket with input value and function name.

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

        function_name = (
            re.search(r"```function=\n(.*?)\n```", response, re.DOTALL).group(1)
            if "```function" in response
            else ""
        )
        if function_name:
            print("I am processing your request, this may take a few seconds ...")
            input = re.search(r"```input=\n(.*?)\n```", response, re.DOTALL).group(1)
            input_values = json.loads(input)
            result = CODE_MAP[function_name](**input_values)
            
            conversation_history.append(
                (text, response)
            )
            text = "I ran the function code and the result is " +  str(result)+". Pretend you ran the function to do the calculation and use this value to provide a short final response to the last question that required function code execution."
            full_context = get_full_context(conversation_history, text)
            return self._chain.invoke(full_context)
        else:
            return response

    def get_architecture(self) -> str:
        return "LLM with RAG KOs and internal code execution with no code sent to LLM"

    def invoke(self, query: str) -> str:
        response = self.process(query, self._conversation_history)

        
        self._conversation_history.append(
            (query, response)
        )  # update history excluding code

        return response
