from .chatbot import Chatbot
from .llm_with_rag_kos_and_external_interpreter import LlmWithRagKosAndExternalInterpreter
from .llm_with_ko_code_tools import LlmWithKoCodeTools
from .llm_with_ko_rag_metadata_and_code_tools import LlmWithKoRagMetadataAndCodeTools
from .plain_llm import PlainLlm
from .plain_llm_assistant import PlainLlmAssistant

chatbot_options = [
    "LlmWithRagKosAndExternalInterpreter",
    "LlmWithKoCodeTools",
    "PlainLlm",
    "PlainLlmAssistant",
    "LlmWithKoRagMetadataAndCodeTools"
]

def init_chatbot_from_str(architecture: str, openai_api_key: str, model_name: str, model_seed: int, knowledge_base: str, embedding_model_name: str, embedding_dimension: int):
    if architecture == "LlmWithRagKosAndExternalInterpreter":
        return LlmWithRagKosAndExternalInterpreter(openai_api_key, model_name, model_seed, knowledge_base)
    elif architecture == "LlmWithKoCodeTools":
        return LlmWithKoCodeTools(openai_api_key, model_name, model_seed, knowledge_base)
    elif architecture == "PlainLlm":
        return PlainLlm(openai_api_key, model_name)
    elif architecture == "PlainLlmAssistant":
        return PlainLlmAssistant(openai_api_key, model_name)
    elif architecture == "LlmWithKoRagMetadataAndCodeTools":
        return LlmWithKoRagMetadataAndCodeTools(openai_api_key, model_name, embedding_model_name, embedding_dimension, knowledge_base)
    else:
        raise ValueError(f"Invalid chatbot architecture {architecture}")

__all__ = [
    "Chatbot",
    "LlmWithRagKosAndExternalInterpreter",
    "LlmWithKoCodeTools",
    "PlainLlm",
    "PlainLlmAssistant",
    "init_chatbot_from_str",
    "chatbot_options"
]