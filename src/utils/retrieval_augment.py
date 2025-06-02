from langchain.retrievers import BM25Retriever
#from langchain_graph_retriever import GraphRetriever
#from graph_retriever.strategies import Eager
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os

def retrieve_similar(docs,vectorstore,query,k):
    
	# Semantic‐search retriever (SS)
	semantic_retriever = vectorstore.as_retriever(
   		search_type="similarity",         # cosine or L2 under the hood
    	search_kwargs={"k": 5}            # return top-5
	)

	# Keyword (BM25) retriever
	keyword_retriever = BM25Retriever.from_documents(
   						 docs,
    					k=k                                # also top-5
	)

	# Graph Retriever ---
	# define which metadata fields to “link” on—e.g., title and source
	# retriever = GraphRetriever(
    # 			store=vectorstore,
	# 			edges=[("title", "title"), ("source", "source")],
	# 			strategy=Eager(k=5, start_k=1, max_depth=2)
	# )

	# Hybrid aggregator
	def hybrid_retrieve(question: str):
		sem_docs   = sem_docs = semantic_retriever.invoke(query)
		key_docs   = keyword_retriever.invoke(query)
		#graph_docs = retriever.get_relevant_documents(query)

		# merge & dedupe (by doc.id or content)
		all_docs = sem_docs + key_docs #+ graph_docs
		seen = set()
		unique = []
		for d in all_docs:
			if d.id not in seen:
				seen.add(d.id)
				unique.append(d)

		# return the top‐k of the merged list
		return unique[:k]

	# prompt to send to the LLM
	prompt = """You are an assistant for question-answering tasks.
    	Use the following pieces of retrieved context to answer the question.
    	If you don't know the answer, search in google  .

    	Question: {question}

    	Context: {context}

    	Answer:
    	"""

	prompt_template = ChatPromptTemplate.from_template(prompt)

	# Plug it into your RunnableParallel-based RAG chain:
	# rag_chain_with_source = RunnableParallel(
	# 	{"context": hybrid_retrieve, "question": RunnablePassthrough()}
	# ).assign(
	# 	answer=(
	# 		RunnablePassthrough.assign(context=lambda x: "\n\n".join(d.page_content for d in x["context"]))
	# 		| ChatPromptTemplate.from_template(prompt_template)
	# 		| ChatGroq(model_name="llama3-70b-8192", streaming=True, groq_api_key=os.getenv("GROQ_API_KEY"))
	# 	)
	# )

	# return rag_chain_with_source

	llm = ChatGroq(
    	model_name="llama3-70b-8192", streaming=True, groq_api_key=os.getenv("GROQ_API_KEY")
)
	"""This code defines a chain where input documents are first formatted,
	then passed through a prompt template,
	and finally processed by an LLM."""

	rag_chain_from_docs = (
		RunnablePassthrough.assign(context=lambda x: "\n\n".join(d.page_content for d in x["context"]))
		| prompt_template
		| llm
	)
	"""This code creates a parallel process:
	one retrieves the context (using a retriever),
	and the other passes the question through unchanged.
	The results are then combined and assigned to the variable `answer` using the `rag_chain_from_docs` processing chain."""

	rag_chain_with_source = RunnableParallel(
		{"context": hybrid_retrieve, "question": RunnablePassthrough()}
	).assign(answer=rag_chain_from_docs)

	return rag_chain_with_source