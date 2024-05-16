from resiliparse import parse
from resiliparse.extract import html2text
import resiliparse
import fasttext
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
import os
from collections import Counter

def extract_text(html_bytes):

    encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
    text = html_bytes.decode(encoding)

    return resiliparse.extract.html2text.extract_plain_text(text)


def identify_language(text, model=None):
    if model is None:
        model = fasttext.load_model("/home/shared/lid.176.bin")
    
    text_processed = text.replace("\n", " ")
    predictions = model.predict(text_processed)

    language = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]

    return language, confidence

def mask_emails(text):
    pattern = r"""([!#-'*+/-9=?A-Z^-~-]+(\.[!#-'*+/-9=?A-Z^-~-]+)*|"([]!#-[^-~ \t]|(\\[\t -~]))+")@([!#-'*+/-9=?A-Z^-~-]+(\.[!#-'*+/-9=?A-Z^-~-]+)*|\[[\t -Z^-~]*])"""

    matches = re.findall(pattern, text)

    # Count the number of matches found
    num_matches = len(matches)

    text_out = re.sub(pattern, "|||EMAIL_ADDRESS|||", text)

    return text_out, num_matches

def mask_phone(text):
    pattern = r"""(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"""

    matches = re.findall(pattern, text)

    # Count the number of matches found
    num_matches = len(matches)

    text_out = re.sub(pattern, "|||PHONE_NUMBER|||", text)

    return text_out, num_matches

def mask_ip(text):
    pattern = r"""((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}"""

    matches = re.findall(pattern, text)

    # Count the number of matches found
    num_matches = len(matches)

    text_out = re.sub(pattern, "|||IP_ADDRESS|||", text)

    return text_out, num_matches

def classify_nsfw(text, nsfw_model=None):
    if nsfw_model is None:
        nsfw_model = fasttext.load_model("/home/shared/dolma-jigsaw-fasttext-bigrams-nsfw.bin")
    
    output = nsfw_model.predict(text)

    return output

def classify_toxic(text, toxic_model=None):
    if toxic_model is None:
        toxic_model = fasttext.load_model("/home/shared/dolma-jigsaw-fasttext-bigrams-hatespeech.bin")
    
    output = toxic_model.predict(text)

    return output

def classify_quality_heuristic(text):
    words = nltk.word_tokenize(text)
    num_words = len(words)

    avg_word_length = sum(len(word) for word in words) / num_words

    #contains an alphabetic character
    alphabetic = [1 for word in words if any(char.isalpha() for char in word)]
    
    if alphabetic != []:
        percentage_alphabetic = sum(alphabetic) / num_words
    else:
        percentage_alphabetic = 0

    #check lines for ending in "..."
    lines = text.split("\n")
    ellipsis = [1 for line in lines if line[-3:] == "..."]

    if ellipsis != []:
        percentage_ellipsis = sum(ellipsis) / len(lines)
    else:
        percentage_ellipsis = 0

    #filter by word count
    if num_words < 50 or num_words > 100000:
        return False
    if avg_word_length < 3 or avg_word_length > 10:
        return False
    if percentage_ellipsis > 0.3:
        return False
    if percentage_alphabetic < 0.8:
        return False
    
    return True


def deduplicate_lines(input_files, output_directory):
    # load all the input files into a list of separated files
    all_lines = []
    # input files is a directory
    
    for file in input_files:
        file_list = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                all_lines.append(hash(line))

    # get count instances of lines
    line_counts = Counter(all_lines)
    # get all hashes which have more than one instance
    duplicate_hashes = [hash for hash, count in line_counts.items() if count > 1]

    # iterate through the files, get their filenames, remove duplicate lines, and write to output directory
    for file in input_files:
        file_list = []
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if hash(line) not in duplicate_hashes:
                    file_list.append(line)
        file_name = os.path.basename(file)
        with open(os.path.join(output_directory, file_name), "w") as f:
            f.writelines(file_list)
            f.close()
    return