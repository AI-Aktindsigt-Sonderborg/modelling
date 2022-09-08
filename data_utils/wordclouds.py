import json
import os

from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

from local_constants import FILTERED_SCRAPE_DIR, DATA_DIR
import matplotlib.pyplot as plt

stopwords_union = set.union(set(stopwords.words('danish')), STOPWORDS)

# danish_stopwords = set(stopwords.words('danish'))
stopwords_union.update(
    ["ved", "kan", "samt", "så", "se", "får", "få", "f eks", "f", "eks", "Stk", "stk", "må", "der",
     "for", "fx", "bl"])


def individual_clouds(in_file: str = 'unique_sentences.json', max_words: int = 75):
    with open(os.path.join(FILTERED_SCRAPE_DIR, in_file), 'r',
              encoding='utf-8') as file:

        all_text = ""
        for i, line in enumerate(file):

            if i == 0:
                init_data_dict = json.loads(line)
                current_muni = init_data_dict['kommune']

            data_dict = json.loads(line)
            new_muni = data_dict['kommune']

            if not new_muni == current_muni:
                wordcloud = WordCloud(stopwords=stopwords_union, max_words=max_words,
                                      background_color="white").generate(
                    all_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Mest benyttede ord i {current_muni}')
                plt.axis("off")
                plt.savefig(f'plots/{current_muni}.png')

                current_muni = new_muni
                all_text = data_dict['text']
            else:
                all_text += " " + data_dict['text']
        wordcloud = WordCloud(stopwords=stopwords_union, max_words=max_words,
                              background_color="white").generate(
            all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Mest benyttede ord i {current_muni}')
        plt.axis("off")
        plt.savefig(f'plots/{current_muni}.png')

def total_cloud(in_file: str = 'unique_sentences.json', max_words: int = 75,
                exclude_muni: str = ''):

    all_text = ""
    with open(os.path.join(DATA_DIR, in_file), 'r',
              encoding='utf-8') as file:
        for i, line in enumerate(file):
            data_dict = json.loads(line)
            if 'Station.Kort' in data_dict['text']:
                print(data_dict['text'])
            if not data_dict['kommune'] == exclude_muni:
                all_text += " " + data_dict['text']

    with open(os.path.join(DATA_DIR, f'all_text_ex-{exclude_muni}.txt'), 'w', encoding='utf-8') as out_file:
        out_file.write(all_text)

    wordcloud = WordCloud(stopwords=stopwords_union, max_words=max_words,
                          background_color="white").generate(
        all_text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Mest benyttede ord i alle kommuner')
    plt.axis("off")
    plt.savefig(f'plots/total_cloud_ex-{exclude_muni}.png')

 def get_vocab_size(in_file: str = 'all_text_ex-.txt'):

        with open(os.path.join(DATA_DIR, in_file), 'r',
                  encoding='utf-8') as file:
            all_text = file.read()

        unique_words = set(all_text.split(' '))
        print(unique_words)
        return unique_words

def make_sentence_boxplots(in_file: str = 'unique_sentences.json'):

    with open(os.path.join(FILTERED_SCRAPE_DIR, in_file), 'r',
              encoding='utf-8') as file:
        lengths = []
        for i, line in enumerate(file):

            if i == 0:
                init_data_dict = json.loads(line)
                current_muni = init_data_dict['kommune']

            data_dict = json.loads(line)
            new_muni = data_dict['kommune']

            if not new_muni == current_muni:

                # plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Mest benyttede ord i {current_muni}')
                plt.axis("off")
                plt.savefig(f'plots/{current_muni}.png')

                current_muni = new_muni
                lengths = [len(data_dict['text'])]
                lengths.append(len())
            else:
                lengths.append(len(data_dict['text']))

        # plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Mest benyttede ord i {current_muni}')
        plt.axis("off")
        plt.savefig(f'plots/{current_muni}.png')

if __name__ == '__main__':

    total_cloud()
    # total_cloud(exclude_muni='kk')
    # individual_clouds()


    get_vocab_size()


    print()
