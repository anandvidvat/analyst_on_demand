

import json
import timeit 

import openai
import chromadb


from llmsherpa.readers import LayoutPDFReader
from llama_index.readers.schema.base import Document
from get_started import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

# getting the configuration key
with open("./config.json","r") as file_obj:
    config = json.loads(file_obj.read())
openai.api_key = config["openai_api_key"]


start_time = timeit.timeit()
print(start_time)



# leveraging llmsherpa package to read the PDF (# https://github.com/nlmatics/llmsherpa#layoutpdfreader)
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
#pdf_url = "https://arxiv.org/pdf/1910.13461.pdf" # also allowed is a file path e.g. /home/downloads/xyz.pdf
pdf_url = "https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)
documents = pdf_reader.read_pdf(pdf_url)





# initialize chromadb path 
# treat it as a database
chroma_client  = chromadb.PersistentClient("./llm_store/chroma_db")
# creating a collection to store the data
chroma_collection = chroma_client.get_or_create_collection("quickstart")
# creating a ChromeVectorStore to store the data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# creating storageContext to process the vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
for chunk in documents.chunks():
    index.insert(Document(text=chunk.to_context_text(), extra_info={}))

# create a query engine with index
query_engine = index.as_query_engine()

query_message = "what do you know about tesla?"

response = query_engine.query(query_message)


print(response)

# testing to ensure that the document can be queried
end_time = timeit.timeit()
print("done")
print(end_time - start_time)