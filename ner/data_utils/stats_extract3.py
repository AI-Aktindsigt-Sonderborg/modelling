import json
import re
from collections import Counter

input_file_name = "user_annotated_output7.jsonl"
output_file_name = "./bilou_user_annotated_output12.jsonl"

common_scandi_misses =  ["/S", "'S", "'s", "´S", "´s", "\"S", "\"s", "”S", "”s", "’s", "’S"]
punctuation_list = [",", ".", ":", "-", "?", "!", "\"", ";", "'", "”", "’", ")", "("]
blacklist_helbred = []
blacklist_forbrydelse = []
wrong_annotation_adresse = ["Sønderborg", "Sønderborg"]
wrong_annotation_person = ["Torben"]
wrong_annotation_org = ["Sønderborg"]


total_sentences = 0
total_entities = []
entity_counter = 0
index_changes = 0
updated_data = []
helbred = []
forbrydelse = []

annotated_data = [] # TEST DATA OVERWRITTEN.
# Open blacklist .json files
with open('blacklist_helbred.json', "r") as h_file:
    blacklist_helbred = json.load(h_file)
with open('blacklist_forbrydelse.json', "r") as f_file:
    blacklist_forbrydelse = json.load(f_file)


# Open output.jsonl file
print(f"reading {input_file_name}")
with open(input_file_name) as f:
    for line in f:
        annotated_data.append(json.loads(line))

bilou_data = []
# For each document in output.jsonl file
for document_data in annotated_data:
    sentence_objects = []
    page_num = 1
    # if text exists
    if document_data['pdf_text']:
        # for each page in document
        for pdf_text in document_data['pdf_text']:
            # if page is longer than 30 characters
            if len(pdf_text) > 30:
                keys_list = list(document_data['text_annotation'])
                if f"page_no:{page_num}" not in keys_list:
                    page_num += 1
                    continue
                page_annotations = document_data['text_annotation'][f'page_no:{page_num}'] # annotations to current page
                current_length = 0
                sentence_num = 1
                # for sentence in pdf page
                for sentence in pdf_text.split("\n\n"):
                    total_sentences += 1
                    # if the sentence is longer/= than 30 characters or less/= than 400 characters
                    if len(sentence) >= 5 and len(sentence) <= 700:
                        reindexed_annotations = []
                        # for annotation in the current page's annotations
                        for annotation in page_annotations:
                            # if annotation fits sentence length add it to a list
                            if annotation['annotation']['state'] != "deleted" and annotation['annotation']['start'] >= current_length and annotation['annotation']['end'] <= (current_length + len(sentence)+2):
                                total_entities.append(annotation['annotation']['annotation'])
                                entity_counter += 1
                                new_annotation = annotation
                                new_annotation['annotation']['start'] -= current_length
                                new_annotation['annotation']['end'] -= current_length
                                reindexed_annotations.append(new_annotation)
                        sentence_objects.append({"sentence":sentence, 'annotations':reindexed_annotations, 'page_number':page_num, 'sentence_number':sentence_num})
                    current_length += len(sentence)+2
                    sentence_num += 1
            page_num += 1
        # for sentence with annotations in list of sentence objects
        for sentence_obj in sentence_objects:
            tags = []
            faulty_annotation = [] # the annotation is faulty if whoever annotated this did not annotate the entire word/entity or a space is missing from the text.
            tagged_sentence = sentence_obj['sentence']
            tagged_sentence = re.sub(r" \n", " ", sentence_obj['sentence'])
            words_sentence = tagged_sentence
            annotations = sorted(sentence_obj['annotations'], key=lambda x: x['annotation']['start'], reverse=True)
            # for annotation in list of sentence annotations
            #print("new sentence")
            current_entities = []
            for annotation in annotations:
                category = annotation['annotation']['annotation']
                annotation_content = annotation['annotation']['content']
                start = annotation['annotation']['start']
                end = annotation['annotation']['end']
                if "\n" in annotation_content:
                    end -= 2
                if (end - start) > 0 and start >= 0 and start < len(tagged_sentence) and end > 0 and end <= len(tagged_sentence):
                    if category == "HELBRED" and annotation_content in wrong_annotation_adresse:
                        category = "ADRESSE"
                    if category == "HELBRED" and annotation_content in wrong_annotation_person:
                        category = "PERSON"
                    if category == "HELBRED" and annotation_content in wrong_annotation_org:
                        category = "ORGANISATION"
                    if category == "HELBRED" and annotation_content in blacklist_helbred:
                        annotation['annotation']['state'] = "deleted"
                        continue
                        faulty_annotation.append(category)
                    if category == "FORBRYDELSE" and annotation_content in blacklist_forbrydelse:
                        annotation['annotation']['state'] = "deleted"
                        faulty_annotation.append(category)
                        continue
                    current_entities.append(category)
                    content_from_sentence = tagged_sentence[start:end]
                    #print(start)
                    if content_from_sentence[0] == " ":
                        content_from_sentence = content_from_sentence[1:]
                        start += 1
                    if content_from_sentence[-1] == " ":
                        content_from_sentence = content_from_sentence[:-1]
                        end -= 1

                    if start-2 >= 0 and start < len(tagged_sentence):
                        if (tagged_sentence[start-1].isalpha() or tagged_sentence[start-1].isdigit()) and not (tagged_sentence[start-2].isalpha() or tagged_sentence[start-2].isdigit()):
                            start -= 1
                    if end > 0 and end+2 < len(tagged_sentence):
                        if (tagged_sentence[end+1].isalpha() or tagged_sentence[end+1].isdigit()) and not (tagged_sentence[end+2].isalpha() or tagged_sentence[end+2].isdigit()):
                            end += 1
                    content_from_sentence = tagged_sentence[start:end]
                    words_sentence = words_sentence[:end] + " " + words_sentence[end:]
                    words_sentence = words_sentence[:start] + " " + words_sentence[start:]
                    annotation_split = re.findall(r'\b\w+\b|[^\w\s]', content_from_sentence, flags=re.UNICODE)
                    # if length of annotation is one word
                    if len(annotation_split) == 1:
                        tagged_sentence = tagged_sentence[:start] + f" U-{category} " + tagged_sentence[end:]
                    # if length of annotation is two words
                    if len(annotation_split) == 2:
                        tagged_sentence = tagged_sentence[:start] + f" B-{category} L-{category} " + tagged_sentence[end:]
                    # if length of annotation is three or more words:
                    if len(annotation_split) >= 3:
                        bilou_string = ""
                        word_count = 0
                        # for word in annotation content. Bascally just adds B- I- and L-
                        for single_anno in annotation_split:
                            word_count +=1
                            if word_count == 1:
                                bilou_string += f" B-{category}"
                            elif word_count >= 1 and word_count != len(annotation_split):
                                bilou_string += f" I-{category}"
                            elif word_count == len(annotation_split):
                                bilou_string += f" L-{category} "
                        tagged_sentence = tagged_sentence[:start] + bilou_string + tagged_sentence[end:]
                else:
                    print("\nFailed Annotation")
                    print(f"sentence {tagged_sentence}")
                    print(f"Category; {category}")
                    print(f"start: {start}")
                    print(f"end: {end}")
                    print(f"sentence len: {len(tagged_sentence)}")
                    faulty_annotation.append(category)
            sentence_obj['annotations'] = annotations
            tagged_sentence = re.sub("\s{2,}", " ", tagged_sentence)
            tagged_tokens = re.findall(r'\b[BILU][-][A-ZÆØÅ]{3,}\b|\b\w+\b|[^\w\s]', tagged_sentence, flags=re.UNICODE)
            words_tokens = re.findall(r'\b\w+\b|[^\w\s]', words_sentence, flags=re.UNICODE)
            tags_tokens = []
            # for words in tagged words if not a BILU-[TAG]
            for tagged_word in tagged_tokens:
                if not re.match(r"[BILU][-][A-ZÆØÅ]{3,}",tagged_word):
                    tagged_word = "O"
                tags_tokens.append(tagged_word)
            if len(words_tokens) == len(tagged_tokens):
                bilou_data.append({"tokens":words_tokens, "tags":tags_tokens, "entities":current_entities, "faulty_annotation": faulty_annotation, "user_annotated":document_data['user_annotated'], "document_id":document_data['document_id'], "page_number":sentence_obj['page_number'], "sentence_num":sentence_obj['sentence_number'], "sentence_data":sentence_obj})
            if len(words_tokens) != len(tagged_tokens):
                print("FAILED LENGTHS")
                print(f"Tags - {len(tagged_tokens)}: {tagged_tokens}")
                print(f"Words - {len(words_tokens)}: {words_tokens}")
                print(f"Sentence: {sentence_obj['sentence']}")
                print(f"Faulty: {faulty_annotation}")
                print(f"Annotations: {sentence_obj['annotations']}")
            wrong_annon_counter = 0
            for annon in sentence_obj['annotations']:
                if annon['annotation']['start'] > len(sentence_obj['sentence']) or annon['annotation']['end'] > len(sentence_obj['sentence']):
                    print(f"Annotation index wrong: {annon}")
                    wrong_annon_counter += 1
print(wrong_annon_counter)
print(f"Sentences:{len(bilou_data)}")
# Create a new jsonl file with list of sentence words and list of bilou tags.

print(f"Total Sentences: {total_sentences}")
my_counts = Counter(total_entities)
for i, c in my_counts.items():
    print(f"entitet '{i}' er blevet annoteret {c} gange")
print(f"Total Entities: {entity_counter}")
print(f"Index changes {index_changes}")
with open(output_file_name, "w", encoding="utf8") as f:
    for bilou_sentence in bilou_data:
        json.dump(bilou_sentence, f, ensure_ascii=False)
        f.write("\n")
print(f"{output_file_name} file saved")
"""
with open(f"./helbred.txt", "w") as hel_f:
    for h in helbred:
        hel_f.write(h + "\n")

with open(f"./forbrydelse.txt", "w") as for_f:
    for fo in forbrydelse:
        for_f.write(fo + "\n")
"""
