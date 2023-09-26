import argparse
import random
from typing import List

from sklearn.model_selection import train_test_split

from ner.data_utils.create_bilou import create_bilou_from_one_document
from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.data_utils.get_dataset import get_dane_train, get_label_list_dane
from ner.data_utils.helpers import map_bilou_to_bio
from ner.local_constants import DATA_DIR, PREP_DATA_DIR
from shared.utils.helpers import read_json_lines, write_json_lines


class NERDataPreprocessing:
    """
    Class to prepare data for NER training.

    :param argparse.Namespace args:
        input arguments from :class: `.DataPrepArgParser`.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @staticmethod
    def create_bilou(args):
        """
        Creates bilou file from raw file.

        :param argparse.Namespace args:
            input arguments from :class: `.DataPrepArgParser`.

        """
        raw_data = read_json_lines(input_dir=DATA_DIR, filename=args.origin_input_file)

        word_tag_mismatch_errors: int = 0
        wrong_index_errors: int = 0
        wrong_raw_index_errors: int = 0
        correct_indexes: int = 0
        entity_data: List[dict] = []
        total_sentences: int = 0
        deleted_annotations: int = 0
        total_indices_reindexed: int = 0
        total_not_danish_counter: int = 0

        for i, obs in enumerate(raw_data):
            print(f"creating bilou from document number {i + 1}")
            single_obs_data, errors = create_bilou_from_one_document(
                input_data=obs, data_number=i
            )
            word_tag_mismatch_errors += errors[0]
            wrong_index_errors += errors[1]
            correct_indexes += errors[2]
            total_sentences += errors[3]
            deleted_annotations += errors[4]
            wrong_raw_index_errors += errors[5]
            total_indices_reindexed += errors[6]
            total_not_danish_counter += errors[7]
            entity_data.extend(single_obs_data)

        write_json_lines(
            out_dir=DATA_DIR, data=entity_data, filename=args.bilou_input_file
        )

        print(f"total valid sentences: {len(entity_data)}")
        print(f"word/tag length mismatch errors: {word_tag_mismatch_errors}")
        print(f"wrong index errors: {wrong_index_errors}")
        print(f"wrong raw index errors: {wrong_raw_index_errors}")
        print(f"Total indices reindexed: {total_indices_reindexed}")
        print(f"deleted annotations: {deleted_annotations}")
        print(f"correct indexes: {correct_indexes}")
        print(f"sentences not danish: {total_not_danish_counter}")
        print(f"total sentences: {total_sentences}")

    @staticmethod
    def filter_entities(args):
        """
        Filters and generates new bilou file from original with only
            specified args.entities
        Parameters
        ----------
        :param argparse.Namespace args: input arguments from :class: `.DataPrepArgParser`.
        -------
        """
        bilou_data = read_json_lines(input_dir=DATA_DIR, filename=args.bilou_input_file)

        out_suffix = "".join([x[0] for x in args.entities])

        for obs in bilou_data:
            obs["tags"] = [
                tag if ((tag[2:] in args.entities) or (tag == "O")) else "O"
                for tag in obs["tags"]
            ]

        write_json_lines(
            out_dir=DATA_DIR, filename="bilou_" + out_suffix, data=bilou_data
        )

        if args.create_bio_file:
            bio_data = bilou_data

            for obs in bio_data:
                obs["tags"] = map_bilou_to_bio(obs["tags"])

            write_json_lines(
                out_dir=DATA_DIR, filename="bio_" + out_suffix,
                data=bio_data
            )

        return bilou_data

    @staticmethod
    def train_val_test_to_json_split(
        args,
        data,
        train_size: float = None,
        test_size: int = None,
        train_outfile: str = None,
        val_outfile: str = None,
        test_outfile: str = None,
        add_dane: bool = False,
        random_seed: int = 2,
    ):
        """
        Read grouped data, split to train, val and test and save json
        :param class_grouped_data: grouped data as list og lists of dicts
        :param train_size: float between 0 and 1 specifying the size of the
        train set where 1 is all data
        :param test_size: int >= 1 specifying the number of sentences in each
        class
        :param train_outfile: if train_outfile specified generate train set
        :param val_outfile: if val_outfile specified generate validation set
        :param test_outfile: if test_outfile specified generate test set
        :return:
        """
        assert train_outfile and test_outfile, (
            "\n At least train_outfile and test_outfile must be specified - "
            "see doc: \n" + NERDataPreprocessing.train_val_test_to_json_split.__doc__
        )

        assert train_size and test_size, (
            "Either train or test size must be specified - see doc: \n"
            + NERDataPreprocessing.train_val_test_to_json_split.__doc__
        )

        if train_outfile and val_outfile and test_outfile and train_size and test_size:
            test_ids = []
            test_data = []
            for entity_i in args.entities:
                entity_i_data = [
                    [i, x]
                    for i, x in enumerate(data)
                    if (entity_i in x["entities"]) and (i not in test_ids)
                ]
                if len(entity_i_data) >= test_size:
                    random.seed(random_seed)  # for reproducability
                    random_selection = random.sample(entity_i_data, test_size)
                    test_ids.extend(x[0] for x in random_selection)
                    test_data.extend(x[1] for x in random_selection)

            train_val = [x for i, x in enumerate(data) if i not in test_ids]

            train_val_split = train_test_split(
                train_val, train_size=train_size, random_state=random_seed
            )

            train = train_val_split[0]
            val = train_val_split[1]

            # Add dane dataset to data for robustness
            if add_dane:
                dane = get_dane_train()
                _, dane_id2label, _, _ = get_label_list_dane()

                for obs in dane:
                    dane_tags = [dane_id2label[x] for x in obs["ner_tags"]]
                    tags = [
                        DataPrepConstants.dane_to_akt_label_mapping[x]
                        for x in dane_tags
                    ]
                    entities = [tag[2:] for tag in tags if "-" in tag]
                    train.append(
                        {
                            "tokens": obs["tokens"],
                            "tags": tags,
                            "sentence": obs["text"],
                            "sentence_anon": "",
                            "doc_id": "",
                            "page_no": "",
                            "sentence_no": 0,
                            "origin_line_no": 0,
                            "entities": entities,
                        }
                    )

                for obs in train:
                    obs["tags"] = map_bilou_to_bio(obs["tags"])


                # val = train_val_split[1]
                for obs in val:
                    obs["tags"] = map_bilou_to_bio(obs["tags"])

                for obs in test_data:
                    obs["tags"] = map_bilou_to_bio(obs["tags"])

                train_outfile = "bio_train"
                val_outfile = "bio_val"
                test_outfile = "bio_test"

            write_json_lines(out_dir=PREP_DATA_DIR, filename=train_outfile, data=train)
            write_json_lines(out_dir=PREP_DATA_DIR, filename=val_outfile, data=val)
            write_json_lines(
                out_dir=PREP_DATA_DIR, filename=test_outfile, data=test_data
            )

        return print("datasets generated")


if __name__ == "__main__":
    prep_parser = DataPrepArgParser()
    prep_args = prep_parser.parser.parse_args()
    # prep_args.bilou_input_file = 'bilou_entities_kasper_all'
    # prep_args.create_bilou = True
    data_prep = NERDataPreprocessing(prep_args)
    # prep_args.add_dane = True

    prep_args.entities = DataPrepConstants.standard_ner_entities

    if prep_args.create_bilou:
        data_prep.create_bilou(args=prep_args)

    bilou = data_prep.filter_entities(prep_args)
    data_prep.train_val_test_to_json_split(
        args=prep_args,
        data=bilou,
        train_size=prep_args.split,
        test_size=prep_args.test_size,
        train_outfile=prep_args.train_outfile,
        val_outfile=prep_args.val_outfile,
        test_outfile=prep_args.test_outfile,
        add_dane=prep_args.add_dane,
    )
