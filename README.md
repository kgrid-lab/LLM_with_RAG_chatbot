# Intelligent Chatbot for Biomedical Knowledge Retrieval and Execution, Using Retrieval-Augmented Generation with LLMs
We implemented a pipeline in a prototype chatbot that leverages LLMs for natural language understanding and retrieval-augmented generation, using biomedical knowledge objects as contexts to answer biomedical questions with expert knowledge. To execute knowledge representations (code) attached to each knowledge object, we integrated an OpenAI Assistant API with a code_interpreter tool into our pipeline. This implementation is more reliable compared to other approaches that do not use a code_interpreter tool, examples of which are available in the archive folder of this repository. Here is an example output for this prototype chatbot:
![example](images/example1.png)

## Environment setup
Make sure poetry is installed
Clone the repository using
```
git clone https://github.com/kgrid-lab/LLM_with_RAG_chatbot.git
```

setup a virtual environment using poetry
```
cd LLM_with_RAG_chatbot
poetry env use 3.11
poetry shell
poetry install
```

Create a .env file in the root of the project and store the following values in it
```
KNOWLEDGE_BASE="KO/"
MODEL="gpt-4o"
EVAL_MODEL="gpt-4o"
MODEL_SEED=1762259501
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```
- The KNOWLEDGE_BASE points to the location of knowledge object context files. Currently "KO" and "KO/clinical_calculators" are supported.
- MODEL defines which OpenAI language model to use. We recommend "gpt-4o" if using an LLM or "gpt-4o-mini" if using an SLM. We currently do not support other vendors.
- EVAL_MODEL defines which OpenAI language model to use for evaluation of language model output.
- MODEL_SEED provides a numerical seed to the OpenAI language model specified by MODEL. Specifying MODEL_SEED is optional. It only affects MODEL, not EVAL_MODEL.
- Get your own API key at [OpenAI's API keys section](https://platform.openai.com/api-keys) and set OPENAI_API_KEY with its value. 
## Run the app 
Once the environment is set up you can run this app from the command line using
```
python llm_with_rag_chatbot/chatbot_cmd.py 
```

To run this app in a web interface, use 
```
python llm_with_rag_chatbot/chatbot_ui.py
``` 
and point your browser to http://localhost:5000