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
     "for"])


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


if __name__ == '__main__':
    def total_cloud(in_file: str = 'unique_sentences.json', max_words: int = 75):

        all_text = ""
        with open(os.path.join(FILTERED_SCRAPE_DIR, in_file), 'r',
                  encoding='utf-8') as file:
            for i, line in enumerate(file):
                data_dict = json.loads(line)
                all_text += " " + data_dict['text']

        wordcloud = WordCloud(stopwords=stopwords_union, max_words=max_words,
                              background_color="white").generate(
            all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Mest benyttede ord i alle kommuner')
        plt.axis("off")
        plt.savefig(f'plots/total_cloud.png')

    total_cloud()

    # with open(os.path.join(DATA_DIR, 'municipality_list'), 'r', encoding='utf-8') as file:
    #     municipalities = file.readlines()

    # sentences = []



    print()
