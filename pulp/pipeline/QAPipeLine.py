from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.sparse import ElasticsearchRetriever
from transformers import QuestionAnsweringPipeline
from ..cy.Reader import TransformersReader as Reader
from pulp.pipeline import MODEL_PATH


class QAPipeLine:
    def __init__(self):
        self.finder = Finder(reader=Reader(
            model_name_or_path=MODEL_PATH,
            tokenizer=MODEL_PATH, use_gpu=0),
            retriever=ElasticsearchRetriever(document_store=ElasticsearchDocumentStore(refresh_type='false')))

    def __call__(self, paper_id: str, question: str):
        results = self.finder.get_answers(question=question, top_k_reader=5, filters={'name': [paper_id]})
        return results
