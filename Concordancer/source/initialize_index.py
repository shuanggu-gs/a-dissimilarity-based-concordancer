import os
import json
from zipfile import ZipFile
from elasticsearch import Elasticsearch
from tqdm import tqdm

def build_elasticsearch(data_path, zipfile_name):
    """
    Initialize elasticsearch and build an index

    :param data_path: string, data path
    :param zipfile_name: string, zipfile's name
    :return:
    """

    with ZipFile(os.path.join(data_path, zipfile_name), 'r') as zip_file:
        filenames = zip_file.namelist()
        root = filenames[0]

        # get corpus id and what is the language used in this corpus, support English and French for now
        with zip_file.open(os.path.join(root, 'corpus.json'), 'r') as f:
            corpus_info = json.load(f)
            corpus_id = corpus_info['id']
            if corpus_info['languages'][0] == 'fr-FR':
                corpus_lang = 'french'
            else:
                corpus_lang = 'english'

        # get all documents' name(id)
        documents_ids = []
        for filename in filenames:
            if filename.startswith(os.path.join(root, 'documents')) and filename.endswith('.json'):
                documents_ids.append(filename.split('/')[-1][:-5])

    assert corpus_id, "Corpus is invalid."
    assert corpus_lang, "Unrecognized corpus language."
    assert documents_ids, "No documents detected."

    done_list = []
    todo_path = os.path.join('bin', zipfile_name + '_todo.json')
    done_path = os.path.join('bin', zipfile_name + '_done.json')
    # check if elasticsearch is set up or not. if not, create two json file
    if not os.path.exists('bin'):
        os.mkdir('bin')
    if not (os.path.exists(todo_path) and os.path.exists(done_path)):
        todo = json.dumps({'document_ids': documents_ids}, indent=4)
        done = json.dumps({'document_ids': done_list}, indent=4)
        with open(todo_path, 'w') as f:
            f.write(todo)
        with open(done_path, 'w') as f:
            f.write(done)
    # if exists, load json files and check if all documents have been processed
    with open(todo_path, 'r') as f:
        todo = json.load(f)
        documents_ids = todo['document_ids']
    with open(done_path, 'r') as f:
        done = json.load(f)
        done_list = done['document_ids']

    # for elastic search, setting up mapping
    mapping = {
        'settings': {
            'index': {
                'number_of_shards': '1',
                'max_result_window': '1000000',
            }
        },
        'mappings': {
            'properties': {
                'id': {'type': 'keyword'},
                'source': {'type': 'text'},
                'title': {'type': 'text'},
                'text': {
                    'type': 'text',
                    'fields': {
                        corpus_lang: {
                            "type": "text",
                            "analyzer": corpus_lang
                        }
                    }
                },
                'language': {'type': 'text'}
            },
        }
    }

    print("Initializing Elasticsearch...")
    es = Elasticsearch()
    if not es.indices.exists(index=corpus_id):
        es.indices.create(
            index=corpus_id,
            body=mapping,
            timeout=60,
            ignore=[400, 404]
        )

    # iterate all documents and create an index for each document
    with ZipFile(os.path.join(data_path, zipfile_name), 'r') as zip_file:
        for i in tqdm(range(len(documents_ids))):
            doc_id = documents_ids[i]
            doc_path = os.path.join(root, 'documents', doc_id + '.json')
            try:
                with zip_file.open(doc_path, 'r') as f:
                    doc = json.load(f)
                    if not es.exists(index=corpus_id, id=doc_id):
                        es.create(index=corpus_id, id=doc_id, body=doc, ignore=[400, 404])

            except:
                print("{} not found!".format(doc_id))
            done_list.append(doc_id)

    # If done, update the status of the done json file
    with open(done_path, 'w') as f:
        done = json.dumps({'document_ids': done_list}, indent=4)
        f.write(done)

    print("Done!")


if __name__ == "__main__":
    # Datasets Names: assnat_slim, covid_slim, assnat, covid
    DATA_PATH = '../incoming'
    FOLDER_NAME = 'assnat_slim.zip'

    build_elasticsearch(data_path=DATA_PATH, zipfile_name=FOLDER_NAME)
