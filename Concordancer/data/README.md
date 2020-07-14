Two datasets used in this project are [**Travaux de l’assemblée nationale**](http://www.assnat.qc.ca/fr/travaux-parlementaires/index.html) for French and [**COVID-19 Open Research Dataset**](https://www.semanticscholar.org/cord19) for English. 

**Travaux de l’assemblée nationale**

Travaux de l’assemblée nationale means Work of the National Assembly. This dataset is derived from literal recordings/transcripts of parliamentary work in Quebec. Parliamentary work is the work, discussion and debate of the Assembly and of parliamentary committees. All of these political documents are in French from 1908 to 2019.
			
**COVID-19 Open Research Dataset**

COVID-19 dataset is built and released by the [Semantic Scholar](https://www.semanticscholar.org/) team at the Allen Institute for AI. It is a free and open resource which is built upon more than 59,000 scholarly articles about the novel coronavirus. 

|datasets|name| language | document counts| token counts | Annotated or not|
|--------|----------|--------------|-----------|----------|------|
|Travaux de l’assemblée nationale| AssNatBase | fr | 5484 | 231,956,633 | yes|
|COVID-19 Open Research Dataset| Covid19Txt |en | 4889 | 295,920,167| yes|

### Dataset Structure

Both datasets are following the same structure as below:
```
- corpus.json
- CorpusStructure.json
    - documents
        - 0a01ca28-7ebc-11ea-b2fa-02420a0000bf.json
        - 0a1e631c-7ebc-11ea-9100-02420a0000bf.json
        - ...
    - groups
        - 757fc491-0f25-4ad7-914c-cd906afa7f13
            - 0a01ca28-7ebc-11ea-b2fa-02420a0000bf.json
            - 0a1e631c-7ebc-11ea-9100-02420a0000bf.json
            - ...
        - 2b830e5f-f607-4ff8-b38e-3a4d028cced8
            - 0a01ca28-7ebc-11ea-b2fa-02420a0000bf.json
            - 0a1e631c-7ebc-11ea-9100-02420a0000bf.json
            - ...
```

*note: 'documents' is storing raw text corpus. Each file is a document. Annotated files are saved under 'group' folder. It should be mentioned that AssNatBase dataset is fully annotated but not Covid19Txt.*

- *corpus.json*
```
# an example
{
    "id": "26188747-d7d5-401d-8414-f48cee2da491",
    "title": "AssNatBase",
    "languages": [
        "fr-FR"
    ],
    "projects": [],
    "creationDate": "2020-04-07T19:56:02.000Z",
    "lastModifiedDate": "2020-04-09T20:42:58.000Z",
    "documentCount": 5897
}
```
*note: this json file is describing the corpus including its unique id, name, languages it covered, creation date, modification date and total number of documents.*

- *CorpusStructure.json*
```
# an example
{
    "buckets": [
        {
            "id": "757fc491-0f25-4ad7-914c-cd906afa7f13",
            "name": "Transcode task bucket",
            "schemas": [
                {
                    "schemaType": "DOCUMENT_META",
                    "jsonSchema": null
                }
            ]
        },
        {
            "id": "2b830e5f-f607-4ff8-b38e-3a4d028cced8",
            "name": "Annotations",
            "schemas": [
                {
                    "schemaType": "term_demo",
                    "jsonSchema": null
                },
                {
                    "schemaType": "SENTENCE",
                    "jsonSchema": null
                },
                {
                    "schemaType": "Timex3",
                    "jsonSchema": null
                },
                {
                    "schemaType": "TOKEN",
                    "jsonSchema": null
                },
                {
                    "schemaType": "TDI_SINGLE",
                    "jsonSchema": null
                }
            ]
        }
    ]
}
```

*note: CorpusStructure explicates the structure of 'groups'. Each bucket corresponds to one folder under 'groups' by matching the id.*

- *Transcode task bucket*
```
# one example file under transcode task bucket
{
    "26188747-d7d5-401d-8414-f48cee2da491": 
    {
        "757fc491-0f25-4ad7-914c-cd906afa7f13": 
        {
            "DOCUMENT_META": 
            [
                {
                    "schemaType": "DOCUMENT_META", 
                    "_documentID": "0a2b03de-7a9b-11ea-8913-02420a0000bf", 
                    "_corpusID": "26188747-d7d5-401d-8414-f48cee2da491", 
                    "file_name": "39-1_20100216.txt", 
                    "file_path": "/brut/39-1/", 
                    "file_type": "text/plain; charset=UTF-8", 
                    "file_encoding": "UTF-8", 
                    "source": "brut.zip", 
                    "indexedLanguage": "fr-FR", 
                    "detectedLanguage": "fr-FR", 
                    "detectedLanguageProb": 99.99974086418732, 
                    "file_creation_date": "2020-04-07T15:43:43Z", 
                    "file_edit_date": "2020-04-02T18:06:48Z", 
                    "document_size": 269574, 
                    "file_size": 269573, 
                    "file_extension": ".txt", 
                    "annotationId": "0a4ff60a-7a9b-11ea-a2f1-02420a0000bf"
                }
            ]
        }
    }
}
```

*note: the first id is corresponding to corpus id and the second id is referring to bucket id. The third id is the schemaType.*

- *Annotations*
```
{
    "26188747-d7d5-401d-8414-f48cee2da491":
    {
        "2b830e5f-f607-4ff8-b38e-3a4d028cced8":
        {
            "SENTENCE":
            [
                {
                    "schemaType": "SENTENCE",
                    "_documentID": "0a2b03de-7a9b-11ea-8913-02420a0000bf",
	                "offsets": [{'begin': 0, 'end': 107}],
                    "string": "Treize heures quarante-six minutes) Le Vice-Président (M. Chagnon): Bon début de semaine, bon mardi matin.",
                    "_corpusID": 26188747-d7d5-401d-8414-f48cee2da491,
                    "length": 107,
                    "annotationId": ba98b970-8542-11ea-8423-02420a00008d
                },
                ...
            ],
            "TOKEN":
            [
                {
                    "_documentID": "0a2b03de-7a9b-11ea-8913-02420a0000bf",
                    "category": "PUN",
                    "string": "(",
                    "offsets": [{'begin': 0, 'end': 1}],
                    "length": 1,
                    "schemaType": "TOKEN",
                    "stem": "(",
                    "lemma": "(",
                    "_corpusID": "26188747-d7d5-401d-8414-f48cee2da491",
                    "annotationId": "baa5a03a-8542-11ea-8ccc-02420a00008d"
                },
                ...
            ]
        }
    }
}
```

*note: the first id is corresponding to corpus id and the second id is referring to bucket id. And two lists are corresponding for two schemaTypes respectively. For SENTENCE schemaType, the annotaion results contain a whole sentence, the start and end indices, length of sentence and so on. For TOKEN schemaType, it is a part-of-speech annotation.*

- *one document json file*
```
{
    "id": "0a2b03de-7a9b-11ea-8913-02420a0000bf",
    "source": "brut.zip",
    "title": "39-1_20100216.txt",
    "text": "(Treize heures quarante-six minutes)
Le Vice-Président (M. Chagnon): Bon début de semaine, bon mardi matin. Veuillez vous asseoir, s'il vous plaît.
Affaires courantes Déclarations de députés
Nous allons passer immédiatement à la rubrique Déclarations des députés, et je vais inviter M. le député de Charlesbourg à prendre la parole. ...",
    "language": "fr-FR"
}
```
*note: a document file containing doc id, source, title, whole context and language.*