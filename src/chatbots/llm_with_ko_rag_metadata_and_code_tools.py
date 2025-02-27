import json
import logging
import os

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from . import Chatbot

# Import KO Python code functions so they can be called directly.
from KO.clinical_calculators.code.ascvd_2013 import ascvd_2013
from KO.clinical_calculators.code.bmi import bmi
from KO.clinical_calculators.code.bsa import bsa
from KO.clinical_calculators.code.chadsvasc import chadsvasc
from KO.clinical_calculators.code.ckd_epi_gfr_2021 import ckd_epi_gfr_2021
from KO.clinical_calculators.code.cockcroft_gault_cr_cl import cockcroft_gault_cr_cl
from KO.clinical_calculators.code.corr_ca_alb import corr_ca_alb
from KO.clinical_calculators.code.mdrd_gfr import mdrd_gfr
from KO.clinical_calculators.code.mean_arterial_pressure import mean_arterial_pressure
import KO.clinical_calculators.code.nihss as nihss
from KO.clinical_calculators.code.wells import wells

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

ENC = "utf-8"

KO_DB_NAME = "ko_db"

CODE_MAP = {
    "ascvd-2013": ascvd_2013,
    "bmi": bmi,
    "bsa": bsa,
    "chadsvasc": chadsvasc,
    "ckd-epi-gfr-2021": ckd_epi_gfr_2021,
    "cockcroft-gault-cr-cl": cockcroft_gault_cr_cl,
    "corr-ca-alb": corr_ca_alb,
    "mdrd-gfr": mdrd_gfr,
    "mean-arterial-pressure": mean_arterial_pressure,
    "nihss": nihss_adapter,
    "wells": wells,
}

PROMPT = """
You are an assistant helping the clinician user perform clinicial calculations.
Follow these steps:
1. Read the user's question to find out if they are requesting you to perform a calculation.
    - If the user is requesting a calculation, identify which function can be used to perform this calculation. If there are multiple functions which can perform this calculation, stop and ask the user which function to use before proceeding.
    - If the user is not requesting a calculation, simply answer their query and do not proceed with the remaining steps.
2. Identify all the parameters of this function.
3. Of these parameters, identify which parameters have not been assigned a value.
4. For each parameter which has not yet been assigned a value:
    - If the parameter is marked [nullable], tell the user this parameter is optional and ask if they would like to provide a value. If they decline, assign this parameter a value of null.
    - If the parameter is not marked [nullable], ask the user to provide a value for this parameter. Once the user provides a value, assign this parameter to the value the user provided.
5. Once all parameters have been assigned values, check the units of these values. If the user provided parameter values in different units than what the function requires, convert these values to the units required by the function.
6. Call the function with the final set of parameter values.
7. Enclose the result of calling the function tool in asterisks and communicate it to the user. (e.g. The patient's creatinine clearance is *75* mL/min.) Mention which function was called.
"""

logger = logging.getLogger(__name__)

def convert_ko_to_tool_metadata(ko_metadata):
    params = ko_metadata["koio:hasKnowledge"]["parameters"]
    return {
        "type": "function",
        "function": {
            "name": ko_metadata["@id"],
            "description": "{}: {}".format(
                ko_metadata["dc:title"], ko_metadata["dc:description"]
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
                "additionalProperties": False,
            },
        },
    }

class LlmWithKoRagMetadataAndCodeTools(Chatbot):
    """
    Through RAG, the LLM has access to a database of the KO metadata files in
    knowledge_base. This allows the LLM to answer queries about the metadata.
    Each of the KO implementation Python functions are registered with the LLM
    as a function tool using the OpenAI Assistants API. This ensures that
    code is executed exactly as written and not subject to hallucination
    by the LLM.
    """

    def __init__(
        self, openai_api_key: str, model_name: str, embedding_model_name: str, embedding_dimension: int, knowledge_base: str
    ):
        """
        Constructor
        Parameters:
        - openai_api_key: See README on how to get one. Recommend putting it in .env.
        - model_name: Which large language model to use. Only OpenAI is currently supported. Recommend storing in .env.
        - embedding_model_name: Which model to use to generate embeddings for RAG. Recommend storing in .env.
        - embedding_dimension: The length (dimensionality) of embeddings. Recommend storing in .env.
        - knowledge_base: knowledge_base/code contains the KO Python code files.
        """

        # Initialize OpenAI client.
        self._client = openai.OpenAI(api_key=openai_api_key)

        # Initialize Qdrant client.
        self._db_client = QdrantClient(location=":memory:")

        # Save certain variables for later use.
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name
        self._embedding_dimension = embedding_dimension

        # Fetch KO metadata JSON files from the knowledge_base.
        ko_metadata_list = []
        for dir_entry in os.scandir(knowledge_base):
            if dir_entry.is_file():
                with open(dir_entry.path, "r", encoding=ENC) as f:
                    ko_metadata_list.append(json.load(f))

        # Process KO metadata into tool metadata in the format required by the OpenAI Assistants API.
        self._tool_metadata_list = [
            convert_ko_to_tool_metadata(ko_metadata) for ko_metadata in ko_metadata_list
        ]

        # Create vector database.
        if not self._db_client.collection_exists(KO_DB_NAME):
            self._db_client.create_collection(
                collection_name=KO_DB_NAME,
                vectors_config=VectorParams(size=self._embedding_dimension, distance=Distance.COSINE),
            )

        # Load KO metadata JSON files into vector database.
        # Payload is file with original formatting.
        # Embeddings are derived from minified JSON.
        # Also include a list of calculators available.
        points = [
            PointStruct(
                id=idx,
                vector = self.create_embedding(json.dumps(ko_metadata)),
                payload = ko_metadata
            )
            for idx, ko_metadata in enumerate(ko_metadata_list)
        ]
        self._db_client.upsert(KO_DB_NAME, points)

        # Build a list of messages for the LLM to respond to.
        # The list will begin with a system message containing the prompt.
        self._messages = [{
            "content": PROMPT,
            "role": "system"
        }]
    
    def create_embedding(self, input: str):
        return self._client.embeddings.create(
            input=input,
            model=self._embedding_model_name
        ).data[0].embedding

    def invoke(self, query: str) -> str:
        # Create an embedding from the query.
        query_embedding = self.create_embedding(query)
        
        # Retrieve the closest-match KO metadata JSON file from the database based on the query embedding.
        hits = self._db_client.search(
            collection_name=KO_DB_NAME,
            query_vector=query_embedding,
            limit=1
        )
        ko_metadata_str = json.dumps(hits[0].payload)

        # RAG: Add a message to the list containing the KO metadata file.
        self._messages.append({
            "content": ko_metadata_str,
            "role": "system"
        })

        # Add the user's query to the end of the message list.
        self._messages.append({
            "content": query,
            "role": "user"
        })

        # Ask the LLM to respond to the messages using the OpenAI Chat Completions API.
        response = self._client.chat.completions.create(
            messages=self._messages,
            model=self._model_name,
            tools=self._tool_metadata_list,
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
        
        # If the LLM is waiting for a function tool to be called,
        # call that function and send the LLM the result.
        elif response_entry.finish_reason == "tool_calls":
            self._messages.append(response_entry.message)
            if len(response_entry.message.tool_calls) != 1:
                logger.error("Unexpected number of tool calls in Chat Completions response {}".format(response))
                return "ERROR"
            tool_call = response_entry.message.tool_calls[0]
            if tool_call.type != "function":
                logger.error("Unexpected tool call type in Chat Completions response {}".format(response))
            func = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            self._messages.append({
                "role": "tool",
                "content": str(CODE_MAP[func](**args)),
                "tool_call_id": tool_call.id
            })
            response_after_tool_call = self._client.chat.completions.create(
                messages=self._messages,
                model=self._model_name,
                tools=self._tool_metadata_list,
                temperature=0
            )

            # Extract the final response from the LLM and return it.
            if len(response_after_tool_call.choices) != 1:
                logger.error("Unexpected number of choice entries in Chat Completions response {}".format(response))
                return "ERROR"
            final_response = response_after_tool_call.choices[0]
            if final_response.finish_reason == "stop":
                self._messages.append(final_response.message)
                return final_response.message.content
            else:
                logger.error("Unexpected status {} in Chat Completions response after tool call {}".format(final_response.finish_reason, final_response))
                return "ERROR"
    
        else:
            logger.error("Unexpected status {} in Chat Completions response after tool call {}".format(response_entry.finish_reason, response_entry))
            return "ERROR"