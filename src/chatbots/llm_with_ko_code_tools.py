import json
import logging
import os

import openai

import KO.clinical_calculators.code.nihss as nihss

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
from KO.clinical_calculators.code.wells import wells

from . import Chatbot

ENC = "utf-8"


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
    "chadsvasc": chadsvasc,
    "ckd-epi-gfr-2021": ckd_epi_gfr_2021,
    "cockcroft-gault-cr-cl": cockcroft_gault_cr_cl,
    "corr-ca-alb": corr_ca_alb,
    "mdrd-gfr": mdrd_gfr,
    "mean-arterial-pressure": mean_arterial_pressure,
    "nihss": nihss_adapter,
    "wells": wells,
}


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


logger = logging.getLogger(__name__)


def get_latest_response(client, thread_id, run_id) -> str:
    """
    Utility function to get the most recent response from an OpenAI assistant.
    """
    messages = client.beta.threads.messages.list(
        thread_id=thread_id, limit=1, order="desc", run_id=run_id
    )
    return messages.data[0].content[0].text.value


class LlmWithKoCodeTools(Chatbot):
    """
    This Chatbot consists of an LLM with each KO Python implementation as a
    registered tool.
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
        - knowledge_base: knowledge_base/code contains the KO Python code files.
        """

        # Fetch KO metadata JSON files from the knowledge_base.
        ko_metadata_list = []
        for dir_entry in os.scandir(knowledge_base):
            if dir_entry.is_file():
                with open(dir_entry.path, "r", encoding=ENC) as f:
                    ko_metadata_list.append(json.load(f))

        # Process KO metadata into tool metadata in the format required by the OpenAI Assistants API.
        tool_metadata_list = [
            convert_ko_to_tool_metadata(ko_metadata) for ko_metadata in ko_metadata_list
        ]

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
        If there are multiple functions which can perform the requested calculation, stop and ask the user which one they would like to use.
Step 2: Identify all the parameters of the function.
Step 3: Identify which of these parameters the user has not provided values for.
Step 4: For each of these parameters, ask the user to provide the value.
        If the parameter can take a value of null, still ask for the value. If the user refuses, then assign a value of null to this parameter.
        If the parameter is not marked [optional], ask the user for the value until they provide it. Do not proceed until the user has provided it. You may need to ask several times.
Step 5: Once values have been obtained for all parameters, check the units.
        If the user provided parameter values in different units than what the function requires, convert these values to the units required by the function.
Step 6: Call the function with the final set of parameter values.
Step 7: Enclose the result of calling the function tool in asterisks and communicate it to the user. (e.g. The patient's creatinine clearance is *75* mL/min.) Mention which function was called.
            """,
            tools=tool_metadata_list,
            model=model_name,
        )

        # Create a thread, which represents a conversation between
        # the clinician and the assistant.
        self._thread = self._client.beta.threads.create()

    def invoke(self, query: str) -> str:
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
        elif run.status == "requires_action":
            # Define the list to store tool outputs
            tool_outputs = []

            # For each tool (i.e. function) call requested by the LLM,
            # call the function here in our native Python environment
            # and enclose the result in asterisks (*) to distinguish it
            # from LLM-provided information.
            for tool in run.required_action.submit_tool_outputs.tool_calls:
                params = json.loads(tool.function.arguments)
                tool_outputs.append(
                    {
                        "tool_call_id": tool.id,
                        "output": str(CODE_MAP[tool.function.name](**params)),
                    }
                )

            # Submit all tool outputs at once after collecting them in a list
            if tool_outputs:
                run_w_tool_outputs = (
                    self._client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=self._thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs,
                    )
                )

                if run_w_tool_outputs.status == "completed":
                    return get_latest_response(
                        self._client, self._thread.id, run_w_tool_outputs.id
                    )
                else:
                    logger.error(
                        "Run with tool outputs failed on {} with status {}\n".format(
                            tool_outputs, run_w_tool_outputs.status
                        )
                    )
                    return ""
            else:
                logger.error("No tool calls!\n")
                return ""
        else:
            logger.error(
                "Run failed on query {} with status {}\n".format(query, run.status)
            )
            return ""
