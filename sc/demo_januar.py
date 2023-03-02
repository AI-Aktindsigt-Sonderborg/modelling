# pylint: skip-file
import os.path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from pyinputplus import inputInt, inputChoice
from scipy.spatial.distance import cosine

from sc.modelling_utils.input_args import SequenceModellingArgParser
from sc.modelling_utils.sequence_classification import SequenceClassification
from shared.data_utils.custom_dataclasses import CosineSimilarity
from shared.utils.helpers import TimeCode, BColors
from sc.local_constants import DATA_DIR

warnings.filterwarnings("ignore")

sc_parser = SequenceModellingArgParser()

code_timer = TimeCode()
label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1,
              'Erhverv og turisme': 2, 'Klima, teknik og miljø': 3,
              'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

top_n = 5

LABELS = list(label_dict)

args = sc_parser.parser.parse_args()

# args.model_name = 'sarnikowski/convbert-small-da-cased'
args.model_name = 'last_model-2022-12-21_10-53-25'
args.labels = LABELS
args.sc_demo = True
args.evaluate_during_training = False
args.load_alvenir_pretrained = True
# args.device = 'cpu'
# args.test_data = 'test_local.json'
# ToDo: Figure out how to deal with max_length
args.max_length = 64

modelling = SequenceClassification(args)

modelling.load_data(train=False, test=True)

# test_data_wrapped = modelling.tokenize_and_wrap_data(data=modelling.test_data)
# test_loader = DataLoader(dataset=test_data_wrapped,
#                          # collate_fn=modelling.data_collator,
#                          batch_size=1,
#                          shuffle=False)

model = modelling.get_model()

# embedding_outputs = modelling.create_embeddings_windowed(
#     model=model)


with open(os.path.join(DATA_DIR, "test_data/test_embeddings"), "rb") as fp:
    embedding_outputs = pickle.load(fp)

X = np.array([x.embedding for x in embedding_outputs]).astype(float)
y_true = [int(modelling.label2id[x.label]) for x in embedding_outputs]
y_pred = [int(modelling.label2id[x.prediction]) for x in embedding_outputs]

predefined_sentences = pd.read_csv(os.path.join(
    DATA_DIR,
    'test_data/semantic_search_examples.csv'),
    sep=';',
    encoding='utf-8').to_dict()

# predefined_labels = list(predefined_sentences.keys())

user_input = inputChoice(
    prompt="Tryk 'y' for at starte tutorial, 'n' for at afslutte:\n",
    choices=['y', 'n'])

if user_input == "n":
    sys.exit(0)

while user_input != "n":

    init_choice_input = inputChoice(
        prompt="skriv selv en sætning ('s') eller vælg fra kataloget ('k')?\n",
        choices=['s', 'k'])

    if init_choice_input == 'k':
        label_prompt = "Vælg en kategori fra kataloget (0-7):\n"
        for i, predefined_label in enumerate(LABELS):
            label_prompt += f'{i}: {predefined_label}\n'

        user_input_label = inputInt(
            prompt=label_prompt, min=0, max=7)
        input_label = LABELS[int(user_input_label)]
        sentence_prompt = f"Vælg en sætning fra {BColors.OKBLUE}" \
                          f"{LABELS[user_input_label]}{BColors.ENDC} " \
                          f"mellem 0 og 3:\n"
        for id in predefined_sentences[LABELS[user_input_label]]:
            sentence_prompt += \
                f'{id}: ' \
                f'{predefined_sentences[LABELS[user_input_label]][id]}\n'

        user_input_sentence_int = inputInt(
            prompt=sentence_prompt, min=0, max=3)

        input_sentence = predefined_sentences[
            LABELS[int(user_input_label)]][int(user_input_sentence_int)]

        del user_input_sentence_int

    elif init_choice_input == 's':
        input_sentence_str = input('Skriv sætning:\n')
        assert isinstance(input_sentence_str, str)
        input_sentence = str(input_sentence_str)
        input_label = ""

        del input_sentence_str

    input_embedding = modelling.predict(
        model=model,
        sentence=input_sentence,
        label=input_label)

    cosine_sims = []
    for i, embedding_output in enumerate(embedding_outputs):
        cosine_sims.append(CosineSimilarity(
            input_sentence=input_embedding.sentence,
            reference_sentence=embedding_output.sentence,
            input_label=input_embedding.label,
            reference_label=embedding_output.label,
            cosine_sim=1 - cosine(input_embedding.embedding,
                                  embedding_output.embedding)
        ))
    top_n_embeddings = sorted(cosine_sims,
                              key=lambda x: x.cosine_sim,
                              reverse=True)[:top_n]

    print(
        f'{BColors.BOLD}Input sætning:{BColors.ENDC} '
        f'{top_n_embeddings[0].input_sentence}\n'
        f'Kategori: {BColors.OKBLUE}{top_n_embeddings[0].input_label}{BColors.ENDC}')

    print(f'{BColors.BOLD}Top {top_n} mest similære sætninger:\n{BColors.ENDC}')

    for i, embedding in enumerate(top_n_embeddings):
        if embedding.cosine_sim < 0.5:
            print_sim_color = BColors.FAIL
        else:
            print_sim_color = BColors.OKGREEN
        print(
            f'{BColors.BOLD}{i + 1}{BColors.ENDC}: {embedding.reference_sentence}\n'
            f'{BColors.BOLD} Kategori: {BColors.ENDC}{BColors.OKBLUE}'
            f'{embedding.reference_label}{BColors.ENDC}\n'
            f'{BColors.BOLD} Semantisk lighed: {BColors.ENDC}'
            f'{print_sim_color}{embedding.cosine_sim}')
        print(BColors.ENDC)
    # Plot pca af et lille udsnit fra hvert kategori
    code_timer.how_long_since_start()

    del init_choice_input

    user_input = inputChoice(
        prompt="Prøv et nyt eksempel?\n Tryk 'y' for ja, 'n' for at afslutte: ",
        choices=['y', 'n'])
