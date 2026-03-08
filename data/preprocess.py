import os
import re

DATASET_PATH = "data/20_newsgroups"


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_dataset():

    documents = []
    labels = []

    categories = os.listdir(DATASET_PATH)

    for label_id, category in enumerate(categories):

        category_path = os.path.join(DATASET_PATH, category)

        if not os.path.isdir(category_path):
            continue

        for file in os.listdir(category_path):

            file_path = os.path.join(category_path, file)

            try:
                with open(file_path, "r", encoding="latin1") as f:
                    text = f.read()
                    text = clean_text(text)

                    documents.append(text)
                    labels.append(label_id)

            except:
                continue

    return documents, labels