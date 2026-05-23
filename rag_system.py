from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

# Load documents manually
files = [
    "telecom_docs/billing_policy.txt",
    "telecom_docs/network_policy.txt",
    "telecom_docs/broadband_guide.txt",
    "telecom_docs/customer_support.txt",
    "telecom_docs/cloud_services.txt"
]

documents = []

for file in files:

    loader = TextLoader(file)

    documents.extend(loader.load())

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = FAISS.from_documents(
    texts,
    embeddings
)

print("RAG system ready")

def retrieve_documents(query):

    results = vectorstore.similarity_search(
        query,
        k=2
    )

    retrieved_text = ""

    for r in results:
        retrieved_text += r.page_content + "\n"

    return retrieved_text

query = "My internet speed is slow"

result = retrieve_documents(query)

print(result)