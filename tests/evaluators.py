"""
Contains Evaluator classes which are used to evaluate chatbot responses according to a rubric.
Each Evaluator class performs the evaluation using a different method.
"""

import evaluate
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

class Evaluator:
    """
    Evaluates chatbot responses.
    """
    def __init__(self):
        self._scores = []

    def get_name(self) -> str:
        """
        Returns the name of the approach/metric used by this Evaluator.
        """
        pass

    def score(self, response: str, exchange: dict) -> float:
        """
        Scores the chatbot response and makes an internal note of its correctness.

        :param response: the chatbot response
        :param exchange: a dict containing the query given to the chatbot and a rubric with which to evaluate the chatbot's response
        """
        pass

    def record_response(self, response: str, exchange: dict) -> None:
        """
        Scores the chatbot response and makes an internal note of its correctness.

        :param response: the chatbot response
        :param exchange: a dict containing the query given to the chatbot and a rubric with which to evaluate the chatbot's response
        """
        self._scores.append(self.score(response, exchange))

    def get_results(self) -> str:
        """
        Returns results of evaluating all chatbot responses.
        """
        points = sum(self._scores)
        total = len(self._scores)
        scores = "\n".join(("Item {}: {}".format(i, self._scores[i]) for i in range(total)))
        return "{} Evaluation Results:\nTotal score {}\n{} points out of {}\nIndividual items:\n{}".format(self.get_name(), points / total, points, total, scores)

class KeywordEvaluator(Evaluator):
    """
    Evaluates chatbot responses based on the presence of keywords specified in the rubric.
    See the file-level docstring for documentation of the "keywords" field of the rubric.
    """
    def score(self, response: str, exchange: dict) -> float:
        keywords = exchange["rubric"]["keywords"]
        for rule in keywords:
            if rule == "containsAll":
                if not all(term.casefold() in response.casefold() for term in keywords[rule]):
                    return 0.0
            elif rule == "containsAny":
                if not any(term.casefold() in response.casefold() for term in keywords[rule]):
                    return 0.0
            else:
                raise ValueError("Invalid field {} found in keywords!".format(rule))
        return 1.0
    
    def get_name(self) -> str:
        return "Keyword"

class BleuEvaluator(Evaluator):
    """
    Evaluates each chatbot response by computing the BLEU
    (Bilingual Evaluation Understudy) score, with order 4,
    between it and the standard.
    """
    def score(self, response: str, exchange: dict) -> float:
        return bleu_metric.compute(predictions=[response], references=[exchange["rubric"]["standard"]])["bleu"]

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
        return rouge_metric.compute(predictions=[response], references=[exchange["rubric"]["standard"]], rouge_types=[self.get_name()])[self.get_name()]

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
        super().__init__()
        self._openai_api_key = openai_api_key
        self._model = model

    def score(self, response: str, exchange: dict) -> float:
        model = ChatOpenAI(openai_api_key=self._openai_api_key, model=self._model)
        prompt = PromptTemplate.from_template(
            """You are grading an agent's responses to questions.
            Question: {question}
            Example of correct response: {standard}
            Agent's response: {response}
            The agent's response is considered correct if it has the same meaning as the example response, even if it is not identical.
            Is the agent's response correct?
            Please answer CORRECT or INCORRECT.
            """
        )
        fields = {
            "question": exchange["query"],
            "standard": exchange["rubric"]["standard"],
            "response": response
        }
        result = model.invoke(prompt.invoke(fields))
        if result.content.lower() == "correct":
            return 1.0
        else:
            return 0.0
        
    def get_name(self) -> str:
        return "LLM"