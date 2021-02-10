from pulp.pipeline.QAPipeLine import QAPipeLine
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

QA_PIPELINE= QAPipeLine()
META_STORE = ElasticsearchDocumentStore(refresh_type='false', index='meta')