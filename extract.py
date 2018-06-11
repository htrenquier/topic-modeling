import os
import email
import re
import csv
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
pronouns = {'i','you','he','she','they','them','we','us','that','which','who','whom','whose','whichever','whoever','whomever'}
verbs = {'would','am','are','will','can','could','may','might'}
months = {'january','february','march','april','may','june','july','august','september','october','novemeber','decemeber',
          'jan','feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov','dec'}
others = {'please','no','for','the','am','pm','and','from'}
names = {'stacey'}
all_stop_words = stop_words.union(pronouns).union(others).union(months).union(verbs).union(names)

def extract_headers(a):
    b = email.message_from_string(a)
    if b.is_multipart():
        for payload in b.get_payload():
            # if payload.is_multipart(): ...
            return payload.get_payload()
    else:
        return b.get_payload()


def rm_regex(a):
    a = re.sub(re_html, "", a)
    a = re.sub(re_egal20, "", a)
    a = re.sub(re_signature, "", a)
    a = re.sub(re_addr_city, "", a)
    a = re.sub(re_email_addr, "", a)
    a = re.sub(re_website_addr, "", a)
    a = re.sub(re_price, "", a)
    a = re.sub(re_tel, "", a)
    a = re.sub(re_date, "", a)
    a = re.sub(re_date2, "", a)
    a = re.sub(re_not_word, "", a)
    a = re.sub(re_special_chars, " ", a)
    a = re.sub(re_rep_cr, "", a)
    a = re.sub(re_rep_space, " ", a)
    return a


def keep_words(a):
    #filter wrong types
    a = re.sub(re_html, "", a)
    a = re.sub(re_egal20, "", a)
    # remove signatures (not proven)
    a = re.sub(re_signature, "", a)
    a = re.sub(re_addr_city, "", a)
    a = re.sub(re_email_addr, "", a)
    a = re.sub(re_website_addr, "", a)
    # extract words and tokenize
    return re.findall(re_word, a)

# regex list
re_word = "[A-Z]?[a-z']+[\-[A-Z]?[a-z]+]*"
re_tel = "\(?[0-9]{3}\)?.[0-9]{3}.[0-9]{4}"
re_Name ="[A-Z][a-z\.]*[ ]?"
re_Names = "("+re_Name+"){2,4}"
re_signature = "("+re_Names+"[\n]{1,3}){2,4}"
re_egal20 ="=20[\s]|=09|\[IMAGE\]|=\n|="
re_rep_cr = "\n\n+"
re_price = "\$\d+"
re_email_addr = ".*@.*\..{2,3}"
re_addr_city = "[A-Z][a-z]*, *[A-Z]{2} *[0-9]{5}"
re_date = "[A-Z][a-z\.]* [0-9]{1,2}, \d{4}"
re_date2 = "\d{1,2}\/\d{1,2}\/\d{1,2}"
re_time = "am|pm|AM|PM"
re_html = "<.*\s?.*>"
re_website_addr = "(https?:\/\/)?.*\.[a-z]{2,4}"
re_not_word = "[a-zA-Z]?[0-9]+[a-zA-Z]?" # eg "8th" or "3RX" or "RT300" or any number
re_special_chars = ",|\.|;|:|--|-\s|\s-|>|<|\)|\(|$|\?|\'|`|&|@|=|!|\[|\]|\*|\+|%|#|\/"
re_rep_space = "  +"

# directory to scan
ScanDir = '/Users/htrenqui/Desktop/mailtest'

k = 0
for root, directories, filenames in os.walk(ScanDir):
    for filename in filenames:
        if(filename == ".DS_Store"):
            continue
        k+=1
        # get email file
        file_path = os.path.join(root, filename)
        file = open(file_path,"r")
        # res file
        doc_name = "res/"+str(k)+".csv"
        content = open(doc_name, "w")
        # remove email headers
        payload = extract_headers(''.join(file.readlines()))
        payload = payload.split("Subject:")[-1]
        # regex filter
        payload_tab = keep_words(payload)
        # lower and stop words filter
        payload_tab_lower = [w.lower() for w in payload_tab if not w.lower() in all_stop_words]
        writer = csv.writer(content)
        writer.writerow(payload_tab_lower)
        content.close()
        file.close()

