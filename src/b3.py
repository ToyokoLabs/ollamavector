from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


PAPER = 's40168-023-01533-x.pdf'
question = 'What is the optimal tool for analyzing bacteriophages within the gut microbiome?'
question = 'According to the data you have, what is the best bioinformatic tool for analyzing bacteriophages for shorter reads within the gut microbiome? Give me just the final answer.'

ollama = Ollama(base_url='http://localhost:11434', model="orca2")
#ollama = Ollama(base_url='http://localhost:11434', model="llama3:70b")
#ollama = Ollama(base_url='http://localhost:11434', model="mistral")
#ollama = Ollama(base_url='http://localhost:11434', model="llama3")

#oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama3")
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

#loader = TextLoader('allofplos/allofplos/starter_corpus/journal.pbio.0020188.xml')
loader = PyPDFLoader(PAPER)
pages = loader.load_and_split()

# data = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
#all_splits = text_splitter.split_documents(data)
all_splits = text_splitter.split_documents(pages)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
#question = "when started the in vitro fertilization techniques?"
#question = "Whoa re the members of the President Bush's Council on Bioethics"
docs = vectorstore.similarity_search(question)
print(len(docs))
print(docs)
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
print(qachain.invoke({"query": question}))
