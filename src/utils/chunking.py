from langchain.text_splitter import RecursiveCharacterTextSplitter
#chunk_size=500: Each chunk will be up to 500 characters long.
#chunk_overlap=20: Each chunk will overlap 20 characters with the next, helping maintain context between chunks.
def chunk_text(full_text,chunk_size, chunk_overlap):
    chunks=[]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   is_separator_regex=False)
    chunks= text_splitter.split_documents(full_text)
    return chunks


