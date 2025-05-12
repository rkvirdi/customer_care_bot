from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os

def retrieve_similar(vectorstore):
# Instantiate a retriever from the vector store
	retriever = vectorstore.as_retriever()

# this formats the docs returned by the retriever
	def format_docs(docs):
		return "\n\n".join(doc.page_content for doc in docs)

# prompt to send to the LLM
	prompt = """You are an assistant for question-answering tasks.
    	Use the following pieces of retrieved context to answer the question.
    	If you don't know the answer, search in google  .

    	Question: {question}

    	Context: {context}

    	Answer:
    	"""

	prompt_template = ChatPromptTemplate.from_template(prompt)

	llm = ChatGroq(
    	model_name="llama3-70b-8192", streaming=True, groq_api_key=os.getenv("GROQ_API_KEY")
)
	"""This code defines a chain where input documents are first formatted,
	then passed through a prompt template,
	and finally processed by an LLM."""

	rag_chain_from_docs = (
		RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
		| prompt_template
		| llm
	)
	"""This code creates a parallel process:
	one retrieves the context (using a retriever),
	and the other passes the question through unchanged.
	The results are then combined and assigned to the variable `answer` using the `rag_chain_from_docs` processing chain."""

	rag_chain_with_source = RunnableParallel(
		{"context": retriever, "question": RunnablePassthrough()}
	).assign(answer=rag_chain_from_docs)

	return rag_chain_with_source