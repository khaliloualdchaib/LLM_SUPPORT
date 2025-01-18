from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import os
import shutil
import pandas as pd
from tqdm import tqdm

def load_documents():
    document_loader = PyPDFLoader("data/Salesforce-CRM-User-Manual-St-Francis-St-Marks.pdf")
    pages = document_loader.load_and_split()
    return pages

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

documents = load_documents()
chunks = split_documents(documents)

CHROMA_PATH = "chroma"

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

db = Chroma.from_documents(
    chunks, get_embedding_function(), persist_directory=CHROMA_PATH
)

PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
        ---
        Answer the question based on the above context: {question}    
        """

def main():
    df = pd.read_csv('Salesforce_Questions.csv')
    data = {"Question": [], "Answer": []}
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Questions"):
        text = row["Question"]
        data["Question"].append(text)
        results = db.similarity_search_with_relevance_scores(text, k=3)
        if len(results) == 0 or results[0][1] < 0.6:
            data["Answer"].append('Unable to find matching results.')
            print(f'Unable to find matching results.')
        else:
            context_text = "\n\n --------------\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=text)
            llm = OllamaLLM(model="llama3.2")
            result = llm.invoke(prompt)
            data["Answer"].append(result)
    res = pd.DataFrame(data)
    res.to_csv("Salesforce_QA.csv")

main()
