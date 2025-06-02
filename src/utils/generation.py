# def generate_answer(user_query,rag_chain_with_source):
# #Retrieves relevant documents (like the first example).
# #Then sends those documents as context to an LLM (Groq/LLaMA-3).
# #The LLM reads the context and generates a natural language answer.
#     response = rag_chain_with_source.invoke(user_query)
#     print(response['answer'].content)
#     print(response['context'][2].page_content)


def generate_answer(user_query, rag_chain_with_source):
    response = rag_chain_with_source.invoke(user_query)
    # Return the answer and the full context
    return response['answer'].content, response['context']

