# Project Introduction
AI based system to check the statment given by user as input is correct or incorrect news by checking various  ranked articles news  fectched via SERP API from trusted source and provide result to user
It is via classical natural language processing , via LLM or via RAG+LLM.
## Watch the demo videos:
classical Natural Language processing system working and results:
https://youtu.be/UBBz8CB2tIk
LLM  system of Natural Language processing system working and results:
https://youtu.be/zt7AclgrMHQ
LLM + RAG  system of Natural Language processing system working and results:
https://youtu.be/AToEbCiQM1Q

## Architecture of project:
![Picture 1](https://github.com/user-attachments/assets/b664d314-cc1a-43e8-9008-6a7534beaaff)



## How to run spyder:
Goto Anacoda Navigator > Environments > nlp_env > Open Terminal
In ther terminal, write "spyder" and enter.

## How to add new dependency to existing environment?
conda activate nlp_env
pip install nltk
Also add "nltk" to requirements.txt

## How to add new Conda Environment?
Add your instructions...
## Prerequisites
1. ollama pull llama3
## Newsapi
NewsAPI.org
NewsApI_Key: 60ff4dd213a441ab9be84c76750b059c

##This app uses Google News via SerpAPI.
SERP_API_KEY = "8198373a9102fdb800c25e0c8337ff05cfce241afeb057f3d5a276588fee86dd"
## Download NLTK tokenizer once.
import nltk
nltk.download("punkt")
##run app via terminal main streamlit app
streamlit run app.py
##directory address before running app
cd /Users/nashitahzainab/Desktop/6Semester/NLP/NewsFactCheck.
##local run via main.app
## Open conda 
conda create -n news-verifier python=3.10
conda create -n nlp_env python=3.11
conda activate nlp_env

## run conda environment
conda activate news-verifier
##Install Dependencies
pip install streamlit pandas scikit-learn nltk requests serpapi
#one time run
python -c "import nltk; nltk.download('punkt')"
#Option 1: Run using Spyder (Anaconda)
#1 Activate environment:
conda activate nlp_env
spyder 
Open main.py
Click Run ▶️
Output will appear in the IPython Console
#Option 2: Run using Terminal
conda activate nlp_env
cd path/to/project-folder: cd /Users/nashitahzainab/Desktop/6Semester/NLP/NewsFactCheck
python main.py
#Academic Use
This project demonstrates applied NLP techniques taught in coursework:
Tokenization
Similarity modeling
Sentiment & stance analysis
Ethical NLP considerations
#News Verifier & Context Analyzer
An interactive fact-checking and news verification web app built with Streamlit, SerpAPI, and Llama 3 (via Ollama).
The app allows users to:
Verify news claims using rule-based NLP analysis
Analyze claims using a local LLM with structured reasoning
Fetch and rank relevant news articles automatically
#🤖 LLM-powered reasoning
Uses Llama 3 via Ollama
Outputs structured JSON verdicts
🖥️ Streamlit UI
Claim input
Two independent actions:
Verify News
Analysis through LLM
# To run app
click anaconda>open terminal> cd /Users/nashitahzainab/Desktop/6Semester/NLP/NewsFactCheck>
streamlit run app.py
#To run llm on terminal directly:
cd /Users/nashitahzainab/Desktop/6Semester/NLP/NewsFactCheck
ollama run llama3
