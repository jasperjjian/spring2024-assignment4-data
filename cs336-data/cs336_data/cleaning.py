from resiliparse import parse
from resiliparse.extract import html2text
import resiliparse
import fasttext
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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

def classify_text(text, nsfw_model_path=None, toxic_model_path=None):

    # Load the tokenizer
    nsfw_tokenizer = AutoTokenizer.from_pretrained(nsfw_model_path)

    # Load the classifier
    nsfw_classifier = AutoModelForSequenceClassification.from_pretrained(nsfw_model_path)

    toxic_tokenizer = AutoTokenizer.from_pretrained(toxic_model_path)

    toxic_classifier = AutoModelForSequenceClassification.from_pretrained(toxic_model_path)

    # Tokenize the text

    nsfw_tokenized = nsfw_tokenizer(text, return_tensors="pt")
    toxic_tokenized = toxic_tokenizer(text, return_tensors="pt")

    # Classify the text
    nsfw_output = nsfw_classifier(**nsfw_tokenized)
    toxic_output = toxic_classifier(**toxic_tokenized)

    print(nsfw_output)
    print(toxic_output)

    return