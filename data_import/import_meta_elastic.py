from transformers import AutoTokenizer, AutoModel

from definitions import from_root_dir
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from pulp.cy.dense.recv import TransformersEmbeddingRetriever


def to_meta_dict(meta: dict) -> dict:
    abs = meta['abstract'] if 'abstract' in meta.keys() else None
    if 'abstract' in meta:
        del meta['abstract']
    return {
        'text': abs,
        'meta': meta
    }


store = ElasticsearchDocumentStore(refresh_type='false', index='meta')
retriever = TransformersEmbeddingRetriever(
    document_store=store,
    embedding_model=AutoModel.from_pretrained(from_root_dir('models/scibert_scivocab_uncased')),
    tokenizer=AutoTokenizer.from_pretrained(from_root_dir('models/scibert_scivocab_uncased'))
)
with open(from_root_dir('data/arxiv-metadata_pickle'), 'rb') as f:
    import pickle

    l = pickle.load(f)


store.write_documents((to_meta_dict(m) for m in l))
store.update_embeddings(retriever,index='meta')
# retriever.embed(['It is shown that, within a Ginzburg-Landau (GL) formalism, the\nsuperconducting fluctuation is insulating at zero temperature even if the\nfluctuation dynamics is metallic (dissipative). Based on this fact, the low\ntemperature behavior of the $H_{c2}$-line and the resistivity curves near a\nzero temperature transition are discussed. In particular, it is pointed out\nthat the neglect of quantum fluctuations in data analysis of the dc resistivity\nmay lead to an under-estimation of the $H_{c2}$ values near zero temperature.\n'])
