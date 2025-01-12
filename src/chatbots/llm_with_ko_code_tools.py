import json
import logging

from dotenv import load_dotenv
import openai

from . import Chatbot

# Import KO Python code functions so they can be called directly.
#load_dotenv()
#sys.path.append(os.path.join(os.getenv("KNOWLEDGE_BASE"), "code"))
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

def nihss_adapter(consciousness: str,
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
          inattention: str) -> int:
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
        nihss.ExtinctionInattention[inattention]
    )

CODE_MAP = {
    "ascvd_2013": ascvd_2013,
    "bmi": bmi,
    "bsa": bsa,
    "chadsvasc": chadsvasc,
    "ckd_epi_gfr_2021": ckd_epi_gfr_2021,
    "cockcroft_gault_cr_cl": cockcroft_gault_cr_cl,
    "corr_ca_alb": corr_ca_alb,
    "mdrd_gfr": mdrd_gfr,
    "mean_arterial_pressure": mean_arterial_pressure,
    "nihss": nihss_adapter,
    "wells": wells
}

CODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ascvd_2013",
            "description": "ASCVD (Atherosclerotic Cardiovascular Disease) 2013 Risk Calculator from AHA/ACC: Estimates the patient's risk of developing their first myocardial infarction or stroke in the next 10 years",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "The patient's age in years. This calculation is only valid for ages 40-75."
                    },
                    "dm": {
                        "type": "boolean",
                        "description": "True if the patient has diabetes. False otherwise."
                    },
                    "sex": {
                        "type": "string",
                        "description": "The patient's sex, either male or female."
                    },
                    "smoker": {
                        "type": "boolean",
                        "description": "True if the patient smokes."
                    },
                    "total_cholesterol": {
                        "type": "integer",
                        "description": "The patient's total serum cholesterol in miligrams per deciliter."
                    },
                    "hdl": {
                        "type": "integer",
                        "description": "The patient's serum high-density lipoprotein cholesterol in miligrams per decileter."
                    },
                    "sbp": {
                        "type": "integer",
                        "description": "The patient's systolic blood pressure in mm Hg."
                    },
                    "htn_tx": {
                        "type": "boolean",
                        "description": "True if the patient is being treated for hypertension. False otherwise."
                    },
                    "race": {
                        "type": "string",
                        "description": "The patient's race, either white or african american."
                    }
                },
                "required": ["age", "dm", "sex", "smoker", "total_cholesterol", "hdl", "sbp", "htn_tx", "race"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bmi",
            "description": "The Body Mass Index (BMI) is computed using the patient's height in meters and weight in kilograms. It can be used to estimate whether a patient is underweight or overweight. 18.5 - 25 is the normal range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "height": {
                        "type": "number",
                        "description": "The patient's height in meters."
                    },
                    "weight": {
                        "type": "number",
                        "description": "The patient's weight in kilograms."
                    }
                },
                "required": ["height", "weight"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bsa",
            "description": "Estimates the patient's body surface area (BSA) in square meters using the Mosteller formula, given the patient's height in centimeters and weight in kilograms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "height": {
                        "type": "number",
                        "description": "The patient's height in centimeters."
                    },
                    "weight": {
                        "type": "number",
                        "description": "The patient's weight in kilograms."
                    }
                },
                "required": ["height", "weight"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chadsvasc",
            "description": "CHA2DS2-VASc Score: A score representing the risk of stroke in a patient with atrial fibrillation. 0 is low-risk in males. 0 or 1 is low-risk in females. All other scores are higher risk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "The patient's age in years."
                    },
                    "sex": {
                        "type": "string",
                        "description": "The patient's sex, male or female."
                    },
                    "chf": {
                        "type": "boolean",
                        "description": "True if the patient has a history of heart failure. False otherwise."
                    },
                    "htn": {
                        "type": "boolean",
                        "description": "True if the patient has a history of hypertension. False otherwise."
                    },
                    "stroke": {
                        "type": "boolean",
                        "description": "True if the patient has a history of stroke, TIA, or thromboembolism. False otherwise."
                    },
                    "vasc": {
                        "type": "boolean",
                        "description": "True if the patient has a history of vascular disease, including prior myocardial infarction, peripheral arterial disease, or aortic plaque. False otherwise."
                    },
                    "dm": {
                        "type": "boolean",
                        "description": "True if the patient has a history of diabetes. False otherwise."
                    }
                },
                "required": ["age", "sex", "chf", "htn", "stroke", "vasc", "dm"]      
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ckd_epi_gfr_2021",
            "description": "Steady-state estimate of glomerular filtration rate (GFR) using 2021 CKD-EPI equations using either creatinine alone or both creatinine and cystatin-C.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sex": {
                        "type": "string",
                        "description": "Sex of the patient, male or female."
                    },
                    "age": {
                        "type": "integer",
                        "description": "Age of the patient in years."
                    },
                    "creatinine": {
                        "type": "number",
                        "description": "Serum creatinine concentration for the patient, in milligrams per deciliter."
                    },
                    "cystatinc": {
                        "type": "number",
                        "description": "[optional] Serum cystatin-c concentration for the patient, in milligrams per liter."
                    }
                },
                "required": ["sex", "age", "creatinine"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cockcroft_gault_cr_cl",
            "description": "Estimates creatinine clearance in adults using the Cockcroft-Gault model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sex": {
                        "type": "string",
                        "description": "Sex of the patient, male or female."
                    },
                    "age": {
                        "type": "integer",
                        "description": "Age of the patient in years."
                    },
                    "weight": {
                        "type": "number",
                        "description": "Weight of the patient in kilograms."
                    },
                    "creatinine": {
                        "type": "number",
                        "description": "Serum creatinine concentration for the patient, in milligrams per deciliter."
                    }
                },
                "required": ["sex", "age", "weight", "creatinine"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "corr_ca_alb",
            "description": "Corrects the patient's measured serum calcium level to account for their serum albumin level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ca": {
                        "type": "number",
                        "description": "The patient's serum calcium level in miligrams per deciliter."
                    },
                    "albumin": {
                        "type": "number",
                        "description": "The patient's serum albumin level in grams per deciliter."
                    },
                    "nl_alb": {
                        "type": "number",
                        "description": "The normal serum albumin level. Optional. The default is 4 mg/dL."
                    }
                },
                "required": ["ca", "albumin"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mdrd_gfr",
            "description": "Estimates the patient's glomerular filtration rate (GFR) using the MDRD equation, given their sex, age, serum creatinine, and race.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sex": {
                        "type": "string",
                        "description": "The patient's sex, either male or female."
                    },
                    "age": {
                        "type": "integer",
                        "description": "The patient's age in years."
                    },
                    "cr": {
                        "type": "number",
                        "description": "The patient's serum creatinine level in miligrams per deciliter."
                    },
                    "race_black": {
                        "type": "boolean",
                        "description": "Optional. True if the patient's race is Black. False otherwise."
                    }
                },
                "required": ["sex", "age", "cr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mean_arterial_pressure",
            "description": "Estimates the mean arterial pressure given the systolic and diastolic pressures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "systolic": {
                        "type": "integer",
                        "description": "The patient's systolic blood pressure in mm Hg."
                    },
                    "diastolic": {
                        "type": "integer",
                        "description": "The patient's diastolic blood pressure in mm Hg."
                    }
                },
                "required": ["systolic", "diastolic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "nihss",
            "description": "The NIH Stroke Scale uses the patient's presenting symptoms and signs to quantify the severity of a suspected stroke.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consciousness": {
                        "type": "string",
                        "enum": [
                            "ALERT_KEENLY_RESPONSIVE",
                            "AROUSES_TO_MINOR_STIMULATION",
                            "REQUIRES_REPEATED_STIMULATION_TO_AROUSE",
                            "MOVEMENTS_TO_PAIN",
                            "POSTURES_OR_UNRESPONSIVE"
                        ],
                        "description": "The patient's response to being asked for their age and the month of the year."
                    },
                    "month_and_age_questions": {
                        "type": "string",
                        "enum": [
                            "BOTH_QUESTIONS_RIGHT",
                            "ONE_QUESTION_RIGHT",
                            "ZERO_QUESTIONS_RIGHT",
                            "DYSARTHRIC_INTUBATED_TRAUMA_OR_LANGUAGE_BARRIER",
                            "APHASIC"
                        ],
                        "description": "The patient's response to being asked for their age and the month of the year."
                    },
                    "blink_eyes_and_squeeze_hands": {
                        "type": "string",
                        "enum": [
                            "PERFORMS_BOTH_TASKS",
                            "PERFORMS_ONE_TASK",
                            "PERFORMS_ZERO_TASKS"
                        ],
                        "description": "The patient's response to being asked to blink their eyes and squeeze the examiner's hand. Can be pantomimed if patient does not understand."
                    },
                    "horizontal_extraocular_movements": {
                        "type": "string",
                        "enum": [
                            "NORMAL",
                            "PARTIAL_GAZE_PALSY_CAN_BE_OVERCOME",
                            "PARTIAL_GAZE_PALSY_CORRECTS_WITH_OCULOCEPHALIC_REFLEX",
                            "FORCED_GAZE_PALSY_CANNOT_BE_OVERCOME"
                        ],
                        "description": "The patient's ability to perform horizontal extraocular movements. (i.e. look left and right with both eyes)"
                    },
                    "visual_fields": {
                        "type": "string",
                        "enum": [
                            "NO_VISUAL_LOSS",
                            "PARTIAL_HEMIANOPIA",
                            "COMPLETE_HEMIANOPIA",
                            "PATIENT_IS_BILATERALLY_BLIND",
                            "BILATERAL_HEMIANOPIA"
                        ],
                        "description": "The integrity of the patient's visual fields."
                    },
                    "facial_palsy": {
                        "type": "string",
                        "enum": [
                            "NORMAL_SYMMETRY",
                            "MINOR_PARALYSIS",
                            "PARTIAL_PARALYSIS",
                            "UNILATERAL_COMPLETE_PARALYSIS",
                            "BILATERAL_COMPLETE_PARALYSIS"
                        ],
                        "description": "The ability of the patient to engage their facial muscles."
                    },
                    "left_arm_motor_drift": {
                        "type": "string",
                        "enum": [
                            "NO_DRIFT_FOR_10_SECONDS",
                            "DRIFTS_BUT_DOES_NOT_HIT_BED",
                            "DRIFTS_HITS_BED",
                            "SOME_EFFORT_AGAINST_GRAVITY",
                            "NO_EFFORT_AGAINST_GRAVITY",
                            "NO_MOVEMENT",
                            "AMPUTATION_JOINT_FUSION"
                        ],
                        "description": "The patient's ability to overcome gravity with their left arm and hold a position without drifting."
                    },
                    "right_arm_motor_drift": {
                        "type": "string",
                        "enum": [
                            "NO_DRIFT_FOR_10_SECONDS",
                            "DRIFTS_BUT_DOES_NOT_HIT_BED",
                            "DRIFTS_HITS_BED",
                            "SOME_EFFORT_AGAINST_GRAVITY",
                            "NO_EFFORT_AGAINST_GRAVITY",
                            "NO_MOVEMENT",
                            "AMPUTATION_JOINT_FUSION"
                        ],
                        "description": "The patient's ability to overcome gravity with their right arm and hold a position without drifting."
                    },
                    "left_leg_motor_drift": {
                        "type": "string",
                        "enum": [
                            "NO_DRIFT_FOR_5_SECONDS",
                            "DRIFTS_BUT_DOES_NOT_HIT_BED",
                            "DRIFTS_HITS_BED",
                            "SOME_EFFORT_AGAINST_GRAVITY",
                            "NO_EFFORT_AGAINST_GRAVITY",
                            "NO_MOVEMENT",
                            "AMPUTATION_JOINT_FUSION"
                        ],
                        "description": "The patient's ability to overcome gravity with their left leg and hold a position without drifting."
                    },
                    "right_leg_motor_drift": {
                        "type": "string",
                        "enum": [
                            "NO_DRIFT_FOR_5_SECONDS",
                            "DRIFTS_BUT_DOES_NOT_HIT_BED",
                            "DRIFTS_HITS_BED",
                            "SOME_EFFORT_AGAINST_GRAVITY",
                            "NO_EFFORT_AGAINST_GRAVITY",
                            "NO_MOVEMENT",
                            "AMPUTATION_JOINT_FUSION"
                        ],
                        "description": "The patient's ability to overcome gravity with their right leg and hold a position without drifting."
                    },
                    "limb_ataxia": {
                        "type": "string",
                        "enum": [
                            "NO_ATAXIA",
                            "ATAXIA_IN_1_LIMB",
                            "ATAXIA_IN_2_LIMBS",
                            "DOES_NOT_UNDERSTAND",
                            "PARALYZED",
                            "AMPUTATION_JOINT_FUSION"
                        ],
                        "description": "The patient's ability to coordinate their limbs."
                    },
                    "sensation": {
                        "type": "string",
                        "enum": [
                            "NORMAL_NO_SENSORY_LOSS",
                            "MILD_MODERATE_LOSS_LESS_SHARP_MORE_DULL",
                            "MILD_MODERATE_LOSS_CAN_SENSE_BEING_TOUCHED",
                            "COMPLETE_LOSS_CANNOT_SENSE_BEING_TOUCHED_AT_ALL",
                            "NO_RESPONSE_AND_QUADRIPLEGIC",
                            "COMA_UNRESPONSIVE"
                        ],
                        "description": "The patients's ability to sense touch."
                    },
                    "language": {
                        "type": "string",
                        "enum": [
                            "NORMAL_NO_APHASIA",
                            "MILD_MODERATE_APHASIA_SOME_OBVIOUS_CHAGNES_WITHOUT_SIGNIFICANT_LIMITATION",
                            "SEVERE_APHASIA_FRAGMENTARY_EXPRESSION_INFERENCE_NEEDED_CANNOT_IDENTIFY_MATERIALS",
                            "MUTE_GLOBAL_APHASIA_NO_USABLE_SPEECH_AUDITORY_COMPREHENSION",
                            "COMA_UNRESPONSIVE"
                        ],
                        "description": "The patient's ability to understand and construct language or lack thereof (i.e. aphasia)."
                    },
                    "dysarthria": {
                        "type": "string",
                        "enum": [
                            "NORMAL",
                            "MILD_MODERATE_DYSARTHRIA_SLURRING_BUT_CAN_BE_UNDERSTOOD",
                            "SEVERE_DYSARTHRIA_UNINTELLIGIBLE_SLURRING_OR_OUT_OF_PROPORTION_TO_DYSPHAGIA",
                            "MUTE_ANARTHRIC",
                            "INTUBATED_UNABLE_TO_TEST"
                        ],
                        "description": "The patient's ability to produce the sounds of language or lack thereof (i.e. dysarthria)."
                    },
                    "inattention": {
                        "type": "string",
                        "enum": [
                            "NO_ABNORMALITY",
                            "VISUAL_TACTILE_AUDITORY_SPATIAL_PERSONAL_INATTENTION",
                            "EXTINCTION_TO_BILATERAL_SIMULTANEOUS_STIMULATION",
                            "PROFOUND_HEMI_INATTENTION",
                            "EXTINCTION_TO_GREATER_THAN_1_MODALITY"
                        ],
                        "description": "The patient's ability to maintain attention or lack thereof."
                    }
                },
                "required": [
                    "consciousness",
                    "month_and_age_questions",
                    "blink_eyes_and_squeeze_hands",
                    "horizontal_extraocular_movements",
                    "visual_fields",
                    "facial_palsy",
                    "left_arm_motor_drift",
                    "right_arm_motor_drift",
                    "left_leg_motor_drift",
                    "right_leg_motor_drift",
                    "limb_ataxia",
                    "sensation",
                    "language",
                    "dysarthria",
                    "inattention"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wells",
            "description": "Wells' Criteria for Pulmonary Embolism: Given features of the patient's history and presenting symptoms and signs, computes a score representing the probability of a non-pregnant adult patient having a pulmonary embolism in the emergency department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clin_sx_dvt": {
                        "type": "boolean",
                        "description": "True if the patient has clinical signs or symptoms of DVT (deep vein thrombosis). False otherwise."
                    },
                    "pe_1_dx": {
                        "type": "boolean",
                        "description": "True if pulmonary embolism is the leading diagnosis or equally likely as another diagnosis. False otherwise."
                    },
                    "hr_gt_100": {
                        "type": "boolean",
                        "description": "True if the patient's heart rate is greater than 100 beats per minute. False otherwise."
                    },
                    "immob_surg": {
                        "type": "boolean",
                        "description": "True if the patient has been immobilized for at least 3 days or has had surgery in the previous 4 weeks. False otherwise."
                    },
                    "prev_dx": {
                        "type": "boolean",
                        "description": "True if the patient has previously been objectively diagnosed with DVT or pulmonary embolism. False otherwise."
                    },
                    "hemoptysis": {
                        "type": "boolean",
                        "description": "True if the patient is experiencing hemoptysis. False otherwise."
                    },
                    "malignancy": {
                        "type": "boolean",
                        "description": "True if the patient has a malignancy that has been treated within the past 6 months or has received palliative care. False otherwise."
                    }
                },
                "required": ["clin_sx_dvt", "pe_1_dx", "hr_gt_100", "immob_surg", "prev_dx", "hemoptysis", "malignancy"]
            }
        }
    }
]

logger = logging.getLogger(__name__)
    
def get_latest_response(client, thread_id, run_id) -> str:
    """
    Utility function to get the most recent response from an OpenAI assistant.
    """
    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
        limit=1,
        order="desc",
        run_id=run_id
        )
    return messages.data[0].content[0].text.value
    
class LlmWithKoCodeTools(Chatbot):
    """
    This Chatbot consists of an LLM with each KO Python implementation as a
    registered tool.
    """

    def __init__(self, openai_api_key: str, model_name: str, model_seed: int, knowledge_base: str):
        """
        Constructor
        Parameters:
        - openai_api_key: See README on how to get one. Recommend putting it in .env.
        - model_name: Which model to use. Only OpenAI is currently supported. Recommend storing in .env.
        - model_seed: Specify seed for reproducibility. Recommend storing in .env.
        - knowledge_base: knowledge_base/code contains the KO Python code files.
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
        Step 2: Gather values for all the required parameters.
                - If the user has not provided values for all the required parameters, please ask them for the missing values. Do not proceed until the user has provided them all. You may need to ask multiple times.
                - Some parameters are optional. Ask the user if they would like to provide values for the optional parameters. Do not proceed until the user has either provided values for the optional parameters or explicitly stated that they do not want to provide values.
                - Sometimes, the user might provide values in different units than what the code requires. In this case, please convert them to the units required by the code.
        Step 3: Once values have been obtained for all required parameters, call the function tool for the requested calculation with the gathered parameter values.
        Step 4: Enclose the result of calling the function tool in asterisks and communicate it to the user. (e.g. The patient's creatinine clearance is *75* mL/min.)
        """,
        tools=CODE_TOOLS,
        model=model_name
        )

        # Create a thread, which represents a conversation between
        # the clinician and the assistant.
        self._thread = self._client.beta.threads.create()

    def invoke(self, query: str) -> str:
        # Add a message to the thread containing the clinician's query.
        self._client.beta.threads.messages.create(
        thread_id=self._thread.id,
        role="user",
        content=query
        )

        # Asking the LLM to respond to the query is an asynchronous task,
        # so we simply wait until it is done (i.e. create_and_poll).
        run = self._client.beta.threads.runs.create_and_poll(
        thread_id=self._thread.id,
        assistant_id=self._assistant.id
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
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": str(CODE_MAP[tool.function.name](**params))
                })

            # Submit all tool outputs at once after collecting them in a list
            if tool_outputs:
                run_w_tool_outputs = self._client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=self._thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )

                if run_w_tool_outputs.status == 'completed':
                    return get_latest_response(self._client, self._thread.id, run_w_tool_outputs.id)
                else:
                    logger.error("Run with tool outputs failed on {} with status {}\n".format(tool_outputs, run_w_tool_outputs.status))
                    return ""
            else:
                logger.error("No tool calls!\n")
                return ""
        else:
            logger.error("Run failed on query {} with status {}\n".format(query, run.status))
            return ""
