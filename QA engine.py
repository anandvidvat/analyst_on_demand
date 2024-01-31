import json,os
import openai
# getting the configuration key
with open("./config.json","r") as file_obj:
    config = json.loads(file_obj.read())

# passing off the OpenAI API Key since by default llama_index used GPT-3.5T under the hood
openai.api_key = config["openai_api_key"]

from llama_index import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.readers import PDFReader
from llama_index.storage import StorageContext
from llama_index.tools import QueryEngineTool
from llama_index.agent import OpenAIAgent

PERSIST_DIR = "./llm_store_pdf/"


agent = OpenAIAgent.from_tools(tools, verbose = True)

update_index = True

if update_index and True : #not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("examples", recursive= True).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k = 3, response_mode="tree_summarize")


# # pass a query to the query_engine for a responce
# response = query_engine.query("What do you know about Anand Vidvat Arza?")
# # printing the responce to stdout
# print(response)
