from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from tqdm import tqdm

document_store = ElasticsearchDocumentStore(refresh_type='false')


def concat_sents(sent):
    i = 0
    ss = ''
    for s in sent:
        i += len(str(s).split())
        ss += s
        if i > 200:
            i = 0
            yield ss
            ss = ''


with open('../data/arxiv-processed-pickle', 'rb') as f:
    import pickle

    dic = pickle.load(f)
for (sents, name) in tqdm(dic):
    to_insert = []
    for s in concat_sents(sents):
        to_insert.append({
            'text': s,
            'meta': {'name': name}
        })
    document_store.write_documents(to_insert)
