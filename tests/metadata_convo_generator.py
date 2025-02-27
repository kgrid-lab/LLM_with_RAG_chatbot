"""Convenience script used to generate metadata_convo.json"""

import json
import os

from dotenv import load_dotenv

ENC = "utf-8"

FIELDS = [
    "@id",
    "koio:kgrid",
    "@type",
    "dc:identifier",
    "version",
    "dc:description",
    "koio:contributors",
]

OUT = "tests/metadata_convo.json"

load_dotenv()
knowledge_base = os.getenv("KNOWLEDGE_BASE")

ko_metadata_list = []
for dir_entry in os.scandir(knowledge_base):
    if dir_entry.is_file():
        with open(dir_entry.path, "r", encoding=ENC) as f:
            ko_metadata_list.append(json.load(f))

query_list = []

for ko_metadata in ko_metadata_list:
    title = ko_metadata["dc:title"]
    for field in FIELDS:
        value = ko_metadata[field]
        query_list.append({
            "query": "What is the {} field of the {} Knowledge Object?".format(field, title),
            "query_categories": ["metadata_field_retrieval"],
            "rubric": {
                "standard": value,
                "keywords": [value],
            },
            "notes": "Chatbot should be able to retrieve specific fields from KOs.",
        })

with open(OUT, "w", encoding=ENC) as f:
    json.dump(query_list, f)