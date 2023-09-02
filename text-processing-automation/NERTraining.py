"""
Aims to prepare data for entity regonition custom model training
"""
import re
import logging
from datetime import datetime
import os
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm


_LOGGER_NAME = "alhassane"
logger = logging.getLogger(_LOGGER_NAME)
logger.set_level(logging.INFO)
TODAY = datetime.today().strftime("%Y%m%d")


class NERTraining:
    """
    Training NER
    """

    def __init__(self):
        """_summary_
        """

        data_path = os.environ.get("PATH")
        entity_tags = os.environ.get("TAGS")
        entity_name = os.environ.get("LABEL")
        output_path = os.environ.get("OUTPUT")
        file_extension = os.environ.get("EXTENSION")

        self.data_path = data_path
        self.file_extension = file_extension
        self.entity_tags = entity_tags
        self.entity_name = entity_name
        self.output_path = output_path

        if file_extension == "xlsx":
            data = pd.read_excel(data_path, engine="openpyxl")
        elif file_extension == "xls":
            data = pd.read_excel(data_path)
        elif file_extension == "csv":
            data = pd.read_csv(data_path)
        self.data = data


    def process_review(self, review):
        """
        split data to tokens
        Args:
            review (str): text

        Returns:
            str: tokens lower form
        """
        processed_token = []
        for token in review.split():
            token = "".join(e.lower() for e in token if e.isalnum())
            processed_token.append(token)
        return " ".join(processed_token)


    def get_unique_tag_entities(self, data):
        """
        This function will get the unique tag names or entities that we want to highlight
        """
        tag_names = set()
        for item in data:
            start_span = item["entities"][0][0]
            end_span = item["entities"][0][1]
            tag_name = item["text"][start_span:end_span]
            # annotations = example['annotations']
            # for dict_item in annotations:
            if tag_name not in tag_names:
                tag_names.add(tag_name)
        return tag_names


    def create_data_for_spacy(
        self, data, entity_tags, entity_name: str, review_column: str = "text"
    ) -> list:
        """
        This function converts the list data into format for spacy binary conversion

        Args:
            data (DataFrame): the dataframe containing the review
            review_column (str): reviews column name
            entity_tags (list): all possible entity tags
            entity_name (str, optional): entity global name. Defaults to 'DRUG'.

        Returns:
            list: list of dictionaries, each containg :
                1. text => the full review
                2. entities => start position, end position and the global name of the entity.
        """
        # Step 1: Let's create the training data
        count = 0
        train_data = []
        for _, item in data.iterrows():
            ent_dict = {}
            review = self.process_review(item[review_column])
            # We will find a drug and its positions once and add to the visited items.
            visited_items = []
            entities = []
            for token in review.split():
                if token in entity_tags:
                    for i in re.finditer(token, review):
                        if token not in visited_items:
                            entity = (i.span()[0], i.span()[1], entity_name)
                            visited_items.append(token)
                            entities.append(entity)
            if len(entities) > 0:
                ent_dict["text"] = review
                ent_dict["entities"] = entities
                train_data.append(ent_dict)
                count += 1

        return train_data


    def split_data_train_dev(self, data):
        """
        split data
        """
        treshold = int((len(data) / 4) * 3)
        train = data[:treshold]
        test = data[treshold:]
        return train, test


    def convert_to_spacy(self, data, output_path):
        """
        This function converts the
        data file to spacy binary files
        by creating a blank nlp model
        and DocBin object
        """
        nlp = spacy.blank("en")
        db_data = DocBin()
        for item in tqdm(data):
            text = item["text"]
            labels = item["entities"]
            # create a doc object from text
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in labels:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents
            db_data.add(doc)
        db_data.to_disk(output_path)


    def main(self)->None:
        """
        Main
        """
        data = self.data
        print("\n---------- Raw Data Example ----------\n")
        print(data[0]["text"], "\n")
        print(f"Number of records:{len(data):.>{10}}")

        print("\n---------- Unique Tag Names ----------\n")
        tag_names = self.get_unique_tag_entities(data)
        print(tag_names, "\n")
        print(f"Number of tags:{len(tag_names):.>{10}}")

        print("\n---------- Final Data Example ----------\n")
        final_data = self.create_data_for_spacy(
            self.data,
            self.entity_tags,
            self.entity_name)
        print(final_data[0])

        print("\n---------- Converting Data ----------\n")
        train, test = self.split_data_train_dev(final_data)
        self.convert_to_spacy(train, f"{self.output_path}/train.spacy")
        self.convert_to_spacy(test, f"{self.output_path}/dev.spacy")


ner = NERTraining()


if __name__ == "__main__":
    ner.main()
