"""
Contains Evaluator classes which are used to evaluate chatbot responses according to a rubric.
Each Evaluator class performs the evaluation using a different method.
"""

import logging

from collections.abc import Mapping, Sequence
import evaluate
import openai


logger = logging.getLogger(__name__)


bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

class EvaluationResult:
    """Contains the results of evaluation."""
    def __init__(self, evaluator_name: str, overall_result: float, category_results: Mapping[str, float], item_results: Sequence[float]):
        self.evaluator_name = evaluator_name
        self.overall_result = overall_result
        self.category_results = category_results
        self.item_results = item_results

    def __str__(self) -> str:
        cat_res_str = "\n".join([f"{cat_name}: {cat_val}" for cat_name, cat_val in self.category_results.items()])
        item_str = "\n".join([f"Item {i}: {val}" for i, val in enumerate(self.item_results)])
        return f"""{self.evaluator_name} results:
        overall_result: {self.overall_result}
        category_results:
        {cat_res_str}
        item_results:
        {item_str}"""

class Evaluator:
    """
    Evaluates chatbot responses.
    """

    def get_name(self) -> str:
        """
        Returns the name of the approach/metric used by this Evaluator.
        """
        pass

    def score(self, response: str, exchange: dict) -> float:
        """
        Scores an individual response from the chatbot

        :param response: the chatbot response
        :param exchange: a dict containing the query given to the chatbot and a rubric with which to evaluate the chatbot's response
        Returns the score, which can range from 0.0 to 1.0
        """
        pass

    def score_conversation(self, responses: list[str], conversation: list[dict]) -> EvaluationResult:
        """
        Scores the chatbot responses as part of the tested conversation.

        :param responses: the chatbot responses
        :param conversations: contains the queries given to the chatbot and rubrics with which to evaluate the chatbot's responses
        Returns the key result as well as a verbose string with more detail.
        """
        # Tabulate the scores.
        total = len(responses)
        if total != len(conversation):
            raise ValueError("Responses and conversation have different lengths.")
        scores = [self.score(responses[i], conversation[i]) for i in range(total)]
        overall_score = sum(scores) / total

        # Generate per-category reporting.
        category_results = {}
        for i in range(total):
            for cat in conversation[i]["query_categories"]:
                if cat not in category_results:
                    category_results[cat] = []
                category_results[cat].append(scores[i])
        category_avg_results = {cat: sum(category_results[cat]) / len(category_results[cat]) for cat in category_results}

        return EvaluationResult(self.get_name(), overall_score, category_avg_results, scores)

class KeywordEvaluator(Evaluator):
    """
    Evaluates chatbot responses based on the presence of keywords specified in the rubric.
    See the file-level docstring for documentation of the "keywords" field of the rubric.
    """
    def score(self, response: str, exchange: dict) -> float:
        keywords = exchange["rubric"]["keywords"]
        
        for keyword in keywords:
            if not self.keyword_match(keyword, response):
                return 0.0
        return 1.0
    
    def keyword_match(self, keyword, response: str) -> bool:
        response_lc = response.casefold()
        if isinstance(keyword, str):
            return keyword.casefold() in response_lc
        elif isinstance(keyword, float):
            if str(keyword) in response_lc:
                return True
            else:
                decimal_places = len(str(keyword)) - str(keyword).find('.') - 1
                for i in range(decimal_places - 1, -1, -1):
                    if str(int(round(keyword, i))) in response_lc:
                        return True
                return False
        elif isinstance(keyword, int):
            return str(keyword) in response_lc
        elif isinstance(keyword, dict):
            if "any" in keyword:
                if isinstance(keyword["any"], list):
                    for keyword_any in keyword["any"]:
                        if self.keyword_match(keyword_any, response):
                            return True
                    return False
            raise ValueError("Invalid object {} found in keywords.".format(keyword))
        else:
            raise ValueError("Invalid object {} found in keywords.".format(keyword))
    
    def get_name(self) -> str:
        return "Keyword"

class BleuEvaluator(Evaluator):
    """
    Evaluates each chatbot response by computing the BLEU
    (Bilingual Evaluation Understudy) score, with order 4,
    between it and the standard.
    """
    def score(self, response: str, exchange: dict) -> float:
        try:
            return bleu_metric.compute(predictions=[response], references=[exchange["rubric"]["standard"]])["bleu"]
        except ZeroDivisionError:
            logger.error("BLEU attempted division by zero when evaluating response:\n{}\nAgainst standard:\n{}\n".format(response, exchange["rubric"]["standard"]))
            return 0

    def get_name(self) -> str:
        return "BLEU"

class RougeEvaluator(Evaluator):
    """
    Evaluates each response by computing the one of the ROUGE
    (Recall-Oriented Understudy for Gisting Evaluation) scores
    between it and the standard.
    """
    def rouge_type(self) -> str:
        pass

    def score(self, response: str, exchange: dict) -> float:
        rouge_t = self.get_name()
        return rouge_metric.compute(predictions=[response], references=[exchange["rubric"]["standard"]], rouge_types=[rouge_t])[rouge_t]

class Rouge1Evaluator(RougeEvaluator):
    """ROUGE-1 (ROUGE with n-grams of length 1)"""
    def get_name(self) -> str:
        return "rouge1"

class Rouge2Evaluator(RougeEvaluator):
    """ROUGE-2 (ROUGE with n-grams of length 2)"""
    def get_name(self) -> str:
        return "rouge2"

class RougeLEvaluator(RougeEvaluator):
    """ROUGE-L (ROUGE using longest common subsequence)"""
    def get_name(self) -> str:
        return "rougeL"

class LlmEvaluator(Evaluator):
    """
    Uses an LLM to assess how close the chatbot's responses are to gold standard responses.
    """
    def __init__(self, openai_api_key: str, model: str):
        # Initialize OpenAI client.
        self._client = openai.OpenAI(api_key=openai_api_key)
        self._model_name = model

    def score(self, response: str, exchange: dict) -> float:
        # Write prompt.
        prompt = f"""You are grading an agent's responses to questions.
            Question: {exchange["query"]}
            Example of correct response: {exchange["rubric"]["standard"]}
            Agent's response: {response}
            The agent's response is considered correct if it has the same meaning as the example response, even if it is not identical.
            Is the agent's response correct?
            Please answer CORRECT or INCORRECT.
            """

        # Ask the LLM to respond to using the OpenAI Chat Completions API.
        response = self._client.chat.completions.create(
            messages=[{"content": prompt, "role": "system"}],
            model=self._model_name,
            temperature=0,
            timeout=120  # Timeout errors were observed with the default timeout.
        )

        # Process the response.
        if len(response.choices) != 1:
            logger.error(f"Unexpected number of choice entries in Chat Completions response {response}")
            return "ERROR"
        response_entry = response.choices[0]

        # If the LLM is done, return the response.
        if response_entry.finish_reason == "stop":
            if response_entry.message.content.lower() == "correct":
                return 1.0
            else:
                return 0.0
        else:
            logger.error(f"Unexpected finish reason in Chat Completions response {response_entry.finish_reason}")
            return 0.0
        
    def get_name(self) -> str:
        return "LLM"