# LLM with RAG chatbot


## Environment setup
!pip install langchain
!pip install python-dotenv
!pip install chromadb
!pip install langchain-community
!pip install langchain-openai
!pip install langchain_chroma
!pip install pypdfloader

? see if these could be replaced with poetry add?

## Implementations
1. **umgpt_chatbot.py**:
This chatbot uses the umgpt toolkit. The deployment of the gpt on the server does not have access to code execution so this chatbot does not produce proper responses.

2. **openai_chatbot.py**:
We used GPT models through OpenAI API to implement this chatbot. This chatbot does not use a code interpreter tool so the LLM tried to iterpret the code calculate the result itself and therefore it usually makes mistakes.

3. **openai_chatbot_with_assistant_api**
We combined GPT models with OpenAI Assistant API to create a pipeline that implements RAG with code execution to run the code available in contexts to answer questions. This implementation is more reliable. Here is an example output for this third model:
    ```
    Enter your query: --> 

    Can you calculate my life year gain if I stop using tobacco considering I am a 65 years old female that has been smoking for 10 years now and I still smoke and I smoke 5 cigarettes a day

    I am processing your request, this may take a few seconds...
    If you stop using tobacco, the calculation suggests that you could gain approximately 5.32 quality-adjusted life years. This value is a rough estimate based on the inputs provided regarding your smoking habits, age, and gender. It serves as an encouragement of the potential health benefits of quitting smoking.

    Enter your query: --> 

    what if I am a male?

    I am processing your request, this may take a few seconds...
    A 65-year-old male who has been smoking for 10 years and currently smokes 5 cigarettes a day would gain approximately 4.94 quality-adjusted life years if he stops smoking.

    Enter your query: --> 

    And if I was 50 years old?

    I am processing your request, this may take a few seconds...
    The life year gain calculated for a 50-year-old male who has been smoking for 10 years and currently smokes 5 cigarettes a day is approximately 6.63 quality-adjusted life years if he stops smoking. This means that by quitting smoking, he could expect to gain an additional 6.63 years of life in good health.
    ```