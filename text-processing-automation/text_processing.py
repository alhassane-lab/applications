"""
Preprocessing text tools based on spacy lib
"""
import re
import pandas as pd
import spacy
from spacy.tokens import Token
from unidecode import unidecode
import PyPDF2


class TextProcessing:
    """
    Aims to clean text ddata
    """

    # liste des produits serviers
    servier_products_list = []

    raw_corpus = ""

    def __init__(self, model, blank=False):
        self.model = model

        if blank is True:
            nlp = spacy.blank(model)
        else:
            nlp = spacy.load(model)
        self.nlp = nlp

    @classmethod
    def model_fr(cls):
        """
        class method that aims to load spacy large french model

        """
        return cls(model="fr_core_news_lg")

    @classmethod
    def model_en(cls):
        """
        class method that aims to load spacy large english model
        """
        return cls(model="en_core_news_lg")

    @classmethod
    def init_corpus(cls, doc) -> None:
        """
        Aims to initialize the corpus
        """
        # build cirpus
        raw_corpus = "".join(doc.values)

        TextProcessing.raw_corpus = raw_corpus

    @classmethod
    def get_product_list(
        cls,
        file_location: str,
        pattern: str = r"(\b[A-Z][A-Z]+|\b[A-Z]\b)",
        non_products: list = None,
        product_name_col: str = None,
        new_product_names: list = None,
    ):
        """
        Aims to update servier product list

        Args:

            file_location (str): csv or pdf file containing products names

            pattern (str, optional): regex to extract product names (to be
            adapte for any situation). Defaults to r"(\b[A-Z][A-Z]+|\b[A-Z]\b)".

            non_products (list, optional): words to remove from the final list. Defaults to None.

            product_name_col (str, optional): product names column in cas u use csv file. Defaults to None.

            new_product_names (list, optional): in case un have a list. Defaults to None.
        """
        products = []
        txt_brut =[]

        # pdf
        if ".pdf" in file_location:
            pdf_file = open(file_location, "rb")
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for i in range(len(pdf_reader.pages)):
                prods = re.findall(pattern, pdf_reader.pages[i].extract_text())
                products.extend(prods)
                txt_brut.append(pdf_reader.pages[i].extract_text())
            # prods = [re.findall(pattern, pdf_reader.pages[i].extract_text()) for i in range(len(pdf_reader.pages))]

        # csv
        elif ".csv" in file_location:
            csv_reader = pd.read_csv(file_location)
            products = csv_reader[product_name_col]

        # list
        else:
            products = new_product_names

        products = [prod.lower() for prod in products if len(prod) > 2]

        # clean non products
        if non_products:
            products_cleaned = [elem for elem in products if elem not in non_products]
        else:
            products_cleaned = products

        TextProcessing.servier_products_list.extend(products_cleaned)

        TextProcessing.servier_products_list = list(
            set(TextProcessing.servier_products_list)
        )
        pdf_file.close()
        print("-" * 40, "\n")
        print("Servier product list france is updated!\n")
        print("-" * 40, "\n")
        print(
            f"Number of products from the list : {len(TextProcessing.servier_products_list):.>{5}}"
        )
        print("-" * 40, "\n")
        return txt_brut


    def process_review(self, review):
        """_summary_

        Args:
            review (str): text

        Returns:
            str: tokens lower form
        """
        processed_token = []
        for token in review.split():
            token = ''.join(e.lower() for e in token if e.isalnum())
            processed_token.append(token)
        return ' '.join(processed_token)


    def clean_text(
        self,
        text: str,
        min_len_word: int = 3,
        regex: str = r"@[\w\d_]+",
        rare_words: list = None,
        force_is_alpha: bool = True,
    ) -> list[str]:
        """
        Aims to process a raw text clean it,
        Args:
            text (str): _description_
            min_len_word (int, optional): _description_. Defaults to 3.
            regex (str, optional): _description_. Defaults to r"@[\w\d_]+".
            rare_words (list, optional): _description_. Defaults to None.
            force_is_alpha (bool, optional): _description_. Defaults to True.

        Returns:
            list[str]: _description_
        """

        # build doc
        doc = self.nlp(text)
        nlp = self.nlp

        # Création de l'attribut personnalisé pour les mentions
        def like_handle(token):
            return re.fullmatch(regex, token.text)

        # like_handle = handle(token)
        # like_handle = lambda token: re.fullmatch(regex, token.text)
        Token.set_extension("like_handle", getter=like_handle, force=True)

        # Ajout des mentions comme exceptions à ignorer lors de la recherche de patterns infixe
        nlp.tokenizer.token_match = re.compile(regex).match

        # Ajout d'un pattern infixe pour découper les mots écrits en CamelCase
        default_infixes = list(nlp.Defaults.infixes)
        default_infixes.append("[A-Z][a-z0-9]+")
        infix_regex = spacy.util.compile_infix_regex(default_infixes)
        nlp.tokenizer.infix_finditer = infix_regex.finditer

        # # Surcharge explicite de certaines exceptions de tokenisation
        # nlp.tokenizer.add_special_case("passe-t-il", [{ORTH: "passe"}, {ORTH: "-"}, {ORTH: "t"}, {ORTH: "-"}, {ORTH: "il"}])
        # nlp.tokenizer.add_special_case("est-ce", [{ORTH: "est"}, {ORTH: "-"}, {ORTH: "ce"}])

        text_clean = [
            unidecode(token.lemma_.lower())
            #unidecode(token.is_stop)
            for token in doc
            if (not token.is_punct)
            and (not token.is_space)
            and (not token.like_url)
            and (not token.is_stop)
            and len(token) > min_len_word
            #and (not token.ent_type_ == "PER")
            and (not token._.like_handle)
        ]

        # only alpha character
        if force_is_alpha:
            alpha_tokens = [w for w in text_clean if w.isalpha()]
        else:
            alpha_tokens = text_clean

        # additionnal stop words
        if rare_words:
            final_tokens = [token for token in alpha_tokens if token not in rare_words]
        else:
            final_tokens = alpha_tokens

        return final_tokens


class EntitiesProcesssing(TextProcessing):
    """
    Aims to observe
    Args:
        TextProcessing (_type_): _description_
    """

    def __init__(self, model, blank=False):
        super().__init__(model, blank)


    def observe_frequency(self)->None:
        """
        Observe tops, uniques et maxx5 words
        """
        corpus = super().raw_corpus
        tokens = super().clean_text(corpus)
        tmp = pd.Series(tokens).value_counts()
        top_words = tmp[:30]
        unique_words = tmp[tmp == 1][:30]
        min_x5_words = tmp[tmp <= 5][:30]
        # print("-" * 120)
        # print(
        #     "| Top words ",
        #     " " * 26,
        #     "|",
        #     "Unique words",
        #     " " * 24,
        #     "|",
        #     "5 words minimum",
        #     " " * 20,
        #     "|",
        # )
        #print("-" * 130)
        for x, xi, y, yi, z, zi in zip(
            top_words[:30].index,
            top_words[:30],
            unique_words[:30].index,
            unique_words[:30],
            min_x5_words[:30].index,
            min_x5_words[:30],
        ):
            print(
                "|",
                f"{x:{30}}{xi:>{10}} ",
                " ",
                "|",
                f"{y:{30}}{yi:>{10}} ",
                " ",
                "|",
                f"{z:{30}}{zi:>{10}}",
                " ",
                "|",
            )
        #print("-" * 130)


    def create_data_for_spacy(self,
                              data,
                              review_column,
                              entity_labels,
                              entity_name='DRUG')->list:
        """_summary_

        Args:
            data (DataFrame): the dataframe containing the review
            review_column (str): reviews column name
            entity_labels (list): entities catgories names
            entity_name (str, optional): entity global name. Defaults to 'DRUG'.

        Returns:
            list: list of dictionaries, each containg :
                1. text => the full review  
                2. entities => start position, end position and the global name of the entity.
        """
        #Step 1: Let's create the training data
        count = 0
        train_data = []
        for _, item in data.iterrows():
            ent_dict = {}
            review = super().process_review(item[review_column])
            # We will find a drug and its positions once and add to the visited items.
            visited_items = []
            entities = []
            for token in review.split():
                if token in entity_labels:
                    for i in re.finditer(token, review):
                        if token not in visited_items:
                            entity = (i.span()[0], i.span()[1], entity_name)
                            visited_items.append(token)
                            entities.append(entity)
            if len(entities) > 0:
                ent_dict['text'] = review
                ent_dict['entities'] = entities
                train_data.append(ent_dict)
                count+=1

        return train_data



    def get_unique_tag_entities(self, spacy_data):
        """
        This function will get the unique tag names or entities that we want to highlight
        """
        tag_names = set()
        for data in spacy_data:
            start_span = data['entities'][0][0]
            end_span = data['entities'][0][1]
            tag_name = data['text'][start_span:end_span]
            #annotations = example['annotations']
            #for dict_item in annotations:
            if tag_name not in tag_names:
                tag_names.add(tag_name)
        return tag_names



    def display_training_data_infoq(self,
                                    data
                                    ) -> None:
        """
        Show infomations about entity traning data

        Args:
            data (list): spacy_training_data
            tag_names (set): unique tag names
        """
        print('\n---------- Raw Data Example ----------\n')
        print(data[0]['text'],"\n")
        tag_names = self.get_unique_tag_entities(data)
        print(f"Number of records:{len(data):.>{10}}")
        print('\n---------- Unique Tag Names ----------\n')
        print(tag_names,"\n")
        print(f"Number of tags:{len(tag_names):.>{10}}")