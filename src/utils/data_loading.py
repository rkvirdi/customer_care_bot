from langchain_community.document_loaders import PyPDFLoader

def load_data(file_paths):
    """
    Given a list of PDF file paths, load them all via PyPDFLoader
    and return a single list of Document objects.
    """
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())   # load() returns a list; extend merges it
    return docs


