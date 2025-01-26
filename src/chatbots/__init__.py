from .chatbot import Chatbot
from .llm_with_rag_kos_and_external_interpreter import LlmWithRagKosAndExternalInterpreter
from .llm_with_ko_code_tools import LlmWithKoCodeTools
from .plain_llm import PlainLlm

def init_chatbot_from_str(architecture: str, openai_api_key: str, model_name: str, model_seed: int, knowledge_base: str):
    if architecture == "LlmWithRagKosAndExternalInterpreter":
        return LlmWithRagKosAndExternalInterpreter(openai_api_key, model_name, model_seed, knowledge_base)
    elif architecture == "LlmWithKoCodeTools":
        return LlmWithKoCodeTools(openai_api_key, model_name, model_seed, knowledge_base)
    elif architecture == "PlainLlm":
        return PlainLlm(openai_api_key, model_name)
    else:
        raise ValueError("Invalid chatbot architecture {}".format(architecture))

__all__ = ["Chatbot", "LlmWithRagKosAndExternalInterpreter", "LlmWithKoCodeTools", "PlainLlm", "init_chatbot_from_str"]