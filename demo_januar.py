import sys
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyinputplus import inputInt, inputChoice
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_utils.custom_dataclasses import CosineSimilarity
from modelling_utils.input_args import SequenceModellingArgParser
from modelling_utils.sequence_classification import SequenceClassification
from utils.helpers import TimeCode, bcolors
from utils.visualization import plot_pca_for_demo

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

embedding_outputs = modelling.create_embeddings_windowed(
    model=model)

X = np.array([x.embedding for x in embedding_outputs]).astype(float)
y_true = [int(modelling.label2id[x.label]) for x in embedding_outputs]
y_pred = [int(modelling.label2id[x.prediction]) for x in embedding_outputs]

predefined_sentences = pd.read_csv(
    'data/test_data/semantic_search_examples.csv',
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
        sentence_prompt = f"Vælg en sætning fra {bcolors.OKBLUE}" \
                          f"{LABELS[user_input_label]}{bcolors.ENDC} " \
                          f"mellem 0 og 3:\n"
        for id in predefined_sentences[LABELS[user_input_label]]:
            sentence_prompt += \
                f'{id}: ' \
                f'{predefined_sentences[LABELS[user_input_label]][id]}\n'

        user_input_sentence_int = inputInt(
            prompt=sentence_prompt, min=0, max=3)

        input_sentence = predefined_sentences[
            LABELS[int(user_input_label)]][int(user_input_sentence_int)]
    elif init_choice_input == 's':
        input_sentence_str = input('Skriv sætning:\n')
        input_sentence = str(input_sentence_str)
        input_label = ""

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
        f'{bcolors.BOLD}Input sætning:{bcolors.ENDC} '
        f'{top_n_embeddings[0].input_sentence}\n'
        f'Kategori: {bcolors.OKBLUE}{top_n_embeddings[0].input_label}{bcolors.ENDC}')

    print(f'{bcolors.BOLD}Top {top_n} mest similære sætninger:\n{bcolors.ENDC}')

    for i, embedding in enumerate(top_n_embeddings):
        if embedding.cosine_sim < 0.5:
            print_sim_color = bcolors.FAIL
        else:
            print_sim_color = bcolors.OKGREEN
        print(
            f'{bcolors.BOLD}{i + 1}{bcolors.ENDC}: {embedding.reference_sentence}\n'
            f'{bcolors.BOLD} Kategori: {bcolors.ENDC}{bcolors.OKBLUE}'
            f'{embedding.reference_label}{bcolors.ENDC}\n'
            f'{bcolors.BOLD} Semantisk lighed: {bcolors.ENDC}'
            f'{print_sim_color}{embedding.cosine_sim}')
        print(bcolors.ENDC)
    # Plot pca af et lille udsnit fra hvert kategori
    code_timer.how_long_since_start()

    user_input = inputChoice(
        prompt="Prøv et nyt eksempel?\n Tryk 'y' for ja, 'n' for at afslutte: ",
        choices=['y', 'n'])
