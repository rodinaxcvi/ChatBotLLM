import nltk
import ssl


## THIS CODE DOWNLOADS WIKIPEDIA PAGES OF DIFFERENT CITIES, AND STORES THEM IN MILVUR TO BE ANALYZED
# BY CHATGPT. ALSO MAKES A CSV OF RESPONSE AND QUERY DATA TO BE USED IN VISUALIZATION TOOLS


'''
The Try-Except-Else statement below is in the original code, and it disables SSL certificate verification
when making HTTPS requests (specifically when downloading from Wikipedia).
 
Secure Sockets Layer is a protocol that ensures secure communication between
a client and a server by encrypting data during transmission.

Certificate verification is an essential part of SSL, as it helps confirm
authenticity of the server and protects against potential man-on-the-middle
attacks. You may need to bypass SSL verification for any reason, but that is your discretion.
''' 


# Try - Except - Else statement. It is left commented out for security reasons but you may need to utilize it.
'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else: 
    ssl._create__default_https_context = _create_unverified_https_context
'''
    


nltk.download("stopwords")
from llama_index import (
    GPTVectorStoreIndex,
    GPTSimpleKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext
)



# Import Langchain, which helps develop apps powered by language models
## Link: https://python.langchain.com/docs/get_started/introduction.html
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import openai




# Get your OpenAI API Key stored as an environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = openai_api_key
# print(openai_api_key)





# Import Milvus, which lets you store, index, and manage embedding vectors 
import pandas as pd 
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server

default_server.start()
vector_store = MilvusVectorStore(
    host = "127.0.0.1",
    port = default_server.listen_port
)



# Set the LLM (Large Language Model) as well as LLM service and storage context.
chatgpt_llm = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=chatgpt_llm)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_transformer = OpenAIEmbedding(llm_predictor=chatgpt_llm, vector_store=vector_store)
vectors_data = []



# Set the titles of Wikipedia pages that you want to scrape
wiki_titles = ["Toronto", "Seattle", "San Francisco", "Chicago", "Boston", "Washington D.C", "Cambridge, Massachusetts", "Houston"]

from pathlib import Path
import requests
for title in wiki_titles:
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext' : True
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    wiki_text = page['extract']

    data_path = Path('data')
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", 'w', encoding='utf-8') as fp:
        fp.write(wiki_text)
    




# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(input_files=[f"data/{wiki_title}.txt"]).load_data()
    
# Build city document index
city_indices = {}
index_summaries = {}

# Add index data to a list that can be used for debugging and visualization tools like Arize-Phoenix
visualization_data = []

for wiki_title in wiki_titles:
    city_indices[wiki_title] = GPTVectorStoreIndex.from_documents(city_docs[wiki_title], service_context=service_context, storage_context=storage_context)
    # Set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"
    index_data = {
        'wiki_title': wiki_title,
        'index_summaries': index_summaries
    }
    visualization_data.append(index_data)


# Composability allows you to to define lower-level indices for each document, and higher-order indices over a collection of documents. To see how this works, 
# imagine defining 1) a tree index for the text within each document, and 2) a list index over each tree index (one document) within your collection.
# Read the docs on composability: https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/composability.html
from llama_index.indices.composability import ComposableGraph
graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50
)


# Query transformations are kind of long to explain so here's the docs on them: https://gpt-index.readthedocs.io/en/v0.6.9/how_to/query/query_transformations.html
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    chatgpt_llm, verbose=True
)

from llama_index.query_engine.transform_query_engine import TransformQueryEngine
custom_query_engines = {}
for index in city_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    transform_metadata = {'index_summary': index.index_struct.summary}
    transformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, transform_metadata=transform_metadata)
    custom_query_engines[index.index_id] = transformed_query_engine

custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
    retriever_mode='simple',
    response_mode='tree_summarize',
    service_context=service_context

)

query_engine_decompose = graph.as_query_engine(
    custom_query_engines=custom_query_engines,)

response_chatgpt = query_engine_decompose.query(
    "Compare and contrast the airports in Seattle, Houston, Toronto. "
)

print(str(response_chatgpt))

custom_query_engines = {}
for index in city_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    custom_query_engines[index.index_id] = query_engine


# Retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/
custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
    retriever_mode='simple',
    response_mode='tree_summarize',
    service_context=service_context
)

query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines
)



## Response from ChatGPT
response_chatgpt = query_engine.query(
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "

)
str(response_chatgpt)

# Gather response data for visualization
response_data = {
    'query': "Compare and contrast the airports in Seattle, Houston, Toronto.",
    'response': str(response_chatgpt)
}
visualization_data.append(response_data)
visualization_df = pd.DataFrame(visualization_data)
visualization_df.to_csv('visualization_data.csv', index=False)
