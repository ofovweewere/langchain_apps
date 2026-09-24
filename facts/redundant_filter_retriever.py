# from langchain.embeddings.base import Embeddings
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
# from langchain.schema import BaseRetriever
from langchain_core.retrievers import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
  embeddings: Embeddings
  chroma: Chroma

  def _get_relevant_documents(self, query):
    # Calculate embeddings for the 'query' string
    emb = self.embeddings.embed_query(query)

    # Take embeddings and feed them into that
    # max_marginal_relevance_search_by_vector
    return self.chroma.max_marginal_relevance_search_by_vector(
      embedding=emb,
      lambda_mult = 0.8
    )

  def get_relevant_documents(self, query):
    # Calculate embeddings for the 'query' string
    emb = self.embeddings.embed_query(query)

    # Take embeddings and feed them into that
    # max_marginal_relevance_search_by_vector
    return self.chroma.max_marginal_relevance_search_by_vector(
      embedding=emb,
      lambda_mult = 0.8
    )

  def aget_relevant_documents(self):
    return []