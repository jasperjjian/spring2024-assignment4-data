from warcio.archiveiterator import ArchiveIterator
from warcio.statusandheaders import StatusAndHeaders
import random
from cs336_data import cleaning
import fasttext

def generate_warc(input_file):
    with open(input_file, 'rb') as input_stream:

        for record in ArchiveIterator(input_stream, arc2warc=True):
            if record.rec_type == 'response' and record.http_headers.statusline.startswith('200'):
                warc_headers = StatusAndHeaders(
                    '200 OK', [('Content-Type', 'text/html')])
                yield record.content_stream().read()


def test_language_id(input_file):
    # get a n=20 random sample of WARCs
    warc_list = []
    for i, f in enumerate(generate_warc(input_file)):
        warc_list.append(f)
        if i == 100:
            break

    warcs = random.sample(warc_list, 20)

    # extract the text

    texts = [cleaning.extract_text(warc) for warc in warcs]
    model = fasttext.load_model("/home/shared/lid.176.bin")
    languages = [cleaning.identify_language(text, model=model) for text in texts]

    return texts, languages


def test_replacements(input_file):
    # sample warcs until 20 replacements of email, phone, and ip are found
    warc_list = []
    replaced = []

    counter = 0
    
    for f in generate_warc(input_file):
        extracted_text = cleaning.extract_text(f)
        masked_phone_text, num_masked_phone = cleaning.mask_phone(extracted_text)
        masked_email_text, num_masked_email = cleaning.mask_emails(masked_phone_text)
        masked_ip_text, num_masked_ip = cleaning.mask_ip(masked_email_text)

        if num_masked_phone + num_masked_email + num_masked_ip > 0:
            warc_list.append(extracted_text)
            replaced.append(masked_ip_text)
            counter += 1
        if counter == 20:
            break
    
    return warc_list, replaced
