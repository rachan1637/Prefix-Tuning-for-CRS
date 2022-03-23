from tensorflow import keras
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os.path
import wget

nltk.download('stopwords')

relations = {'Brother', 'Sister', 'Son', 'Daughter', 'Mother', 'Father', 'Boyfriend', 'Girlfriend', 'Aunt', 'Uncle',
             'Nephew', 'Niece', 'Grandmother', 'Grandfather', 'Grandson', 'Granddaughter', 'Stepson', 'Stepdaughter',
             'Stepsister', 'Stepbrother', 'Brother', 'Father', 'Sister', 'Mother-in-law', 'Fiance', 'Fiancee',
             'Husband', 'Wife', 'Ex-wife', 'Husband', 'Boyfriend', 'Girlfriend'}

locations = {"Spring lake", "Saintjohns", "Riverside", "Mercy drive", "Allandale", "Charlestown", "Lake nona south",
             "New malibu", "Oakdale", "Lake nona central", "Franklinto", "Lancaster park", "Barbershop", "York mills",
             "Massachusetts ave", "Tri-south", "Mit", "Everett", "Rocky-fork blacklick accord", "Cambridge highland",
             "West cambridge", "Annex", "School", "Mission hill", "Waterfront", "Southside", "City center", "Peabody",
             "Police station", "Fort columbus airport", "Rosedale", "Montopolis", "Chelsea", "Saint johns",
             "Old west austin", "City of orlando", "Kenton", "Sunderland", "Fenway-kenmore", "Dental office",
             "Windsor road", "East boston", "Brewery", "Downtown", "Bridlemile", "Crestview", "Forest park",
             "Strawberry hill", "Wellington-harrington", "Martin luther king-hwy 183", "Mid cambridge", "Airport",
             "Supermarket", "Alameda", "Lawrence park north", "Oakville", "Far southwest", "Roosevelt park",
             "Computer lab", "Morningside park", "Washington shores", "Longwood medical area", "North central",
             "Homestead", "Harbor islands", "Fashion studio", "Area iv", "Cabbage town", "Highland", "Milo grogan",
             "The north end", "Thorncliffee park", "The southwest", "Malibu groves", "West columbus interim",
             "Parkrose", "Kleinburg", "Ross island", "Main square", "Memorial dr", "Pharmacy", "Dancing studio",
             "Old town-chinatown", "Mattapan", "Parker lane", "Olentangy river road", "Central", "Lloyd",
             "Upper arlington", "Agassiz-harvard", "Eastmoreland", "Office", "Cambridgeport", "Clinic", "Parma court",
             "The west end", "Hyde park", "Georgian acres", "Aliston", "Forest hills", "East cambridge", "Markham",
             "Construction site", "Fenway", "Yorkville", "Roxbury", "Farm", "Worthington", "Orwin manor", "Bridgeton",
             "Southwest hills", "Carver shores", "University", "Sunnybrook", "Chemical lab", "Centralsqaure",
             "Virginia-highland", "Signal hill", "North burnett", "North orange", "Far north", "Mall", "Clintonville",
             "Old cambridge", "Dorchester", "East congress", "Law office", "Johnston terrace", "Institution",
             "Rose isle", "Back bay", "Mill park", "Candler park", "Flemingdon park", "Black creek",
             "Convenience store", "South eola", "Fresh pond pky", "Inman park", "The south end", "Gateway",
             "Music studio", "South river city", "Crescent town", "Powellhurst", "Rowena gardens", "North cambridge",
             "Beacon hill", "Lake nona", "Near southside", "Allston", "Johnson village", "Midtown", "Harrison west",
             "Revere", "Oakridge", "Thorncliffe park", "Lynn", "East somerville", "Bexley", "Jamaica plain", "Zilker",
             "Hillsdale", "The south side", "Jewelry store", "Ormewood park", "West roxbury", "Lake sunset",
             "Scarborough city centre", "Buckhead", "Hospital", "Mlk", "Linnton", "South linden", "Barton hills",
             "Bank", "Centennial", "Poncey highland", "Corktown", "The seaport district"}

if not os.path.isfile("names.txt"):
    wget.download("https://raw.githubusercontent.com/rbouadjenek/SIGIR22_LMRec/main/names.txt")
names = set(line.strip() for line in open('names.txt'))
stopwords = stopwords.words('english')
nltk.download('punkt')


class Dataset:
    def __init__(self, dataset, masking=False):
        self.dataset = dataset
        self.masking = masking
        dataset_path = self.data_path()
        print('Data in: ', dataset_path)
        df = pd.read_csv(dataset_path, lineterminator='\n')
        df = df[['business_id', 'review_text', 'name', 'categories']]
        df.dropna(subset=['business_id', 'review_text', 'name', 'categories'], inplace=True)
        self.df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.item_dist = self.get_item_dist()
        input_ids_list, attention_mask_list, y, labels = self.encode_data()
        self.num_labels = len(labels)
        train_size = int(len(y) * 0.8)
        eval_size = int(len(y) * 0.1)

        self.labels = labels
        self.X_train = [np.array(input_ids_list[0:train_size]), np.array(attention_mask_list[0:train_size])]
        self.y_train = np.array(y[0:train_size])

        self.X_eval = [np.array(input_ids_list[train_size:train_size + eval_size]),
                       np.array(attention_mask_list[train_size:train_size + eval_size])]
        self.y_eval = np.array(y[train_size:train_size + eval_size])

        self.X_test = [np.array(input_ids_list[train_size + eval_size:]),
                       np.array(attention_mask_list[train_size + eval_size:])]
        self.y_test = np.array(y[train_size + eval_size:])
        # Extract categories
        self.business_to_categories = self.get_business_to_categories()

    def encode_data(self):
        labels = []
        input_ids_list = []
        attention_mask_list = []
        y = []
        data = self.df.values.tolist()
        # Save the slow pretrained tokenizer
        slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        save_path = "bert_base_uncased/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        slow_tokenizer.save_pretrained(save_path)

        # Load the fast tokenizer from saved file
        tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
        with tqdm(total=len(data)) as pbar:
            for v in data:
                pbar.update(1)
                item = v[0]
                review_text = v[1]
                name = v[2]
                # check if enough review for the item
                if self.item_dist[item] < self.min_reviews:
                    continue
                # mask item name from the review
                mask = ' '.join(['[MASK]' for x in range(len(name.split()))])
                pattern = re.compile(name, re.IGNORECASE)
                review_text = pattern.sub(mask, review_text)
                if self.masking:
                    review_text = self.mask_entities(review_text)
                # Process text
                tokenized_review = tokenizer.encode(review_text)
                input_ids = tokenized_review.ids
                attention_mask = [1] * len(input_ids)
                # Pad and create attention masks.
                # Truncate if needed
                padding_length = self.max_len - len(input_ids)
                if padding_length > 0:  # pad
                    input_ids = input_ids + ([0] * padding_length)
                    attention_mask = attention_mask + ([0] * padding_length)
                else:
                    input_ids = input_ids[0:self.max_len]
                    input_ids[-1] = 102
                    attention_mask = attention_mask[0:self.max_len]
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                # Process labels
                if item not in labels:
                    labels.append(item)
                y.append(labels.index(item))
        return input_ids_list, attention_mask_list, y, labels

    def get_item_dist(self):
        item_dist = {}
        data = self.df.values.tolist()
        with tqdm(total=len(data)) as pbar:
            for v in data:
                pbar.update(1)
                item = v[0]
                # Process text
                if item in item_dist:
                    v = item_dist[item]
                    item_dist[item] = v + 1
                else:
                    item_dist[item] = 1
        return item_dist

    def data_path(self):
        if self.dataset == 'atlanta':
            data_url = 'http://206.12.93.90:8080/yelp_data/Atlanta_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Atlanta_reviews.csv', data_url, untar=True)
            self.min_reviews = 70
            self.max_len = 400
        elif self.dataset == 'austin':
            data_url = 'http://206.12.93.90:8080/yelp_data/Austin_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Austin_reviews.csv', data_url, untar=True)
            self.min_reviews = 60
            self.max_len = 400
        elif self.dataset == 'boston':
            data_url = 'http://206.12.93.90:8080/yelp_data/Boston_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Boston_reviews.csv', data_url, untar=True)
            self.min_reviews = 115
            self.max_len = 500
        elif self.dataset == 'cambridge':
            data_url = 'http://206.12.93.90:8080/yelp_data/Cambridge_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Cambridge_reviews.csv', data_url, untar=True)
            self.min_reviews = 60
            self.max_len = 400
        elif self.dataset == 'columbus':
            data_url = 'http://206.12.93.90:8080/yelp_data/Columbus_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Columbus_reviews.csv', data_url, untar=True)
            self.min_reviews = 50
            self.max_len = 400
        elif self.dataset == 'orlando':
            data_url = 'http://206.12.93.90:8080/yelp_data/Orlando_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Orlando_reviews.csv', data_url, untar=True)
            self.min_reviews = 65
            self.max_len = 400
        elif self.dataset == 'portland':
            data_url = 'http://206.12.93.90:8080/yelp_data/Portland_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Portland_reviews.csv', data_url, untar=True)
            self.min_reviews = 50
            self.max_len = 400
        else:
            data_url = 'http://206.12.93.90:8080/yelp_data/Toronto_reviews.tar.gz'
            dataset_path = keras.utils.get_file('Toronto_reviews.csv', data_url, untar=True)
            self.min_reviews = 100
            self.max_len = 400
        return dataset_path

    def get_business_to_categories(self):
        business_to_categories = {}
        for business_id, categories in zip(self.df['business_id'], self.df['categories']):
            categories = self.multireplace(categories)
            categories = set(categories.lower().split(', '))
            if 'restaurants' in categories:
                categories.remove('restaurants')
            if business_id in business_to_categories:
                categories = categories | business_to_categories[business_id]
            business_to_categories[business_id] = categories
        return business_to_categories

    def multireplace(self, string, replacements={'/': ',', '(': ',', ')': ''}):
        """
        Given a string and a replacement map, it returns the replaced string.
        :param str string: string to execute replacements on
        :param dict replacements: replacement dictionary {value to find: value to replace}
        :rtype: str
        """
        # Place longer ones first to keep shorter substrings from matching
        # where the longer ones should take place
        # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against
        # the string 'hey abc', it should produce 'hey ABC' and not 'hey ABc'
        substrs = sorted(replacements, key=len, reverse=True)

        # Create a big OR regex that matches any of the substrings to replace
        regexp = re.compile('|'.join(map(re.escape, substrs)))

        # For each match, look up the new string in the replacements
        return regexp.sub(lambda match: replacements[match.group(0)], string)

    def mask_entities(self, aText_string, exlusion_list=[]):
        exlusion_list = [t.lower() for t in exlusion_list]
        tokenized_text = word_tokenize(aText_string)
        token_txt = set([txt.capitalize() for txt in tokenized_text if txt.lower() not in stopwords
                         and txt.lower() not in exlusion_list])

        token_txt = token_txt.intersection(names.union(relations, locations))
        for txt in token_txt:
            aText_string = re.compile(re.escape(txt), re.IGNORECASE).sub('[MASK]', aText_string)
        return aText_string


if __name__ == '__main__':
    Dataset = Dataset('toronto')