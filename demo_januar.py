from typing import List

import pandas as pd
from termcolor import colored, cprint
from utils.helpers import TimeCode, bcolors
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader

from data_utils.custom_dataclasses import CosineSimilarity
from modelling_utils.input_args import SequenceModellingArgParser
from modelling_utils.sequence_classification import SequenceClassification

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
# args.max_length = 512
modelling = SequenceClassification(args)

modelling.load_data(train=False, test=True)

test_data_wrapped = modelling.tokenize_and_wrap_data(data=modelling.test_data)
test_loader = DataLoader(dataset=test_data_wrapped,
                         # collate_fn=modelling.data_collator,
                         batch_size=1,
                         shuffle=False)

model = modelling.get_model()

embedding_outputs = modelling.create_embeddings(
    data_loader=test_loader,
    model=model)

predefined_embeddings = pd.read_csv(
    'data/test_data/semantic_search_examples.csv',
    sep=';',
    encoding='utf-8').to_dict()

predefined_labels = list(predefined_embeddings.keys())

user_input = input('Tryk j for at starte tutorial: ')

while user_input != "s":
    for i, predefined_label in enumerate(predefined_labels):
        print(f'{i}: {predefined_label}\n')
    user_input_label = input(
        "Vælg en kategori fra kataloget (0-7) ('s' for at stoppe):\n")

    user_input_sentence = input("Vælg en sætning mellem 0 og 3:\n")

    if (0 <= int(user_input_label) <= 7) and (0 <= int(user_input_sentence) <=3):
        input_sentence = predefined_embeddings[
            predefined_labels[int(user_input_label)]][int(user_input_sentence)]

    input_embedding = modelling.predict(
        model=model,
        sentence=input_sentence,
        label=predefined_labels[int(user_input_label)-1])

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
        f'{bcolors.BOLD}Input sætning:{bcolors.ENDC} {top_n_embeddings[0].input_sentence}\n')
    print(f'{bcolors.BOLD}Top {top_n} mest similære sætninger:\n{bcolors.ENDC}')

    for i, embedding in enumerate(top_n_embeddings):
        print(
            f'{bcolors.BOLD} {i} {bcolors.ENDC}: {embedding.reference_sentence}\n '
            f'{bcolors.BOLD} Kategori: {bcolors.ENDC}{bcolors.OKBLUE}{embedding.reference_label}\n')
        print(bcolors.ENDC)
    # Plot pca af et lille udsnit fra hvert kategori
    code_timer.how_long_since_start()

    user_input = input("Start forfra? - 'j' for ja, 's' for at stoppe):\n")
