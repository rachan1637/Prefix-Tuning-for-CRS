from csv import unregister_dialect
from tensorflow import keras
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast, BertTokenizerFast, BartTokenizerFast
from tokenizers import BertWordPieceTokenizer
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os.path
import wget
from dataclasses import dataclass
from typing import List
from ast import literal_eval
# nltk.download('stopwords')

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

# if not os.path.isfile("names.txt"):
#     wget.download("https://raw.githubusercontent.com/rbouadjenek/SIGIR22_LMRec/main/names.txt")
# names = set(line.strip() for line in open('names.txt'))
# stopwords = stopwords.words('english')
# nltk.download('punkt')


class Dataset:
    def __init__(self, model_type, csv_file = None, masking=False, max_len=400, reindex=True):
        if max_len != None:
            self.max_len = max_len
        self.masking = masking
        if csv_file is None:
            dataset_path = self.data_path()
            print('Data in: ', dataset_path)
            df = pd.read_csv(dataset_path, lineterminator='\n')
        else:
            df = pd.read_csv(csv_file, engine = 'python')

        if csv_file is None:
            df = df[['business_id', 'review_text', 'name', 'categories']]
        else:
            df = df[['business_id', 'review_text', 'name', 'categories', 'keyphrase_list', 'user_id']]
        df.dropna(subset=['business_id', 'review_text', 'name', 'categories'], inplace=True)
        if reindex:
            self.df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        else:
            self.df = df
        self.item_dist = self.get_item_dist()
        input_ids_list, attention_mask_list, y, labels, keyphrase_ids_list, attention_mask_key_list, user_labels, user_id_list = \
            self.encode_data(model_type = model_type)
        self.num_labels = len(labels)
        self.train_size = int(len(y) * 0.8)
        self.eval_size = int(len(y) * 0.1)

        self.labels = labels
        self.X_train = [np.array(input_ids_list[0:self.train_size]), np.array(attention_mask_list[0:self.train_size])]
        self.y_train = np.array(y[0:self.train_size])

        self.X_eval = [np.array(input_ids_list[self.train_size:self.train_size + self.eval_size]),
                       np.array(attention_mask_list[self.train_size:self.train_size + self.eval_size])]
        self.y_eval = np.array(y[self.train_size:self.train_size + self.eval_size])

        self.X_test = [np.array(input_ids_list[self.train_size + self.eval_size:]),
                       np.array(attention_mask_list[self.train_size + self.eval_size:])]
        self.y_test = np.array(y[self.train_size + self.eval_size:])

        self.X_key_train = [
            np.array(keyphrase_ids_list[: self.train_size]), 
            np.array(attention_mask_key_list[: self.train_size])
        ]
        self.X_key_eval = [
            np.array(keyphrase_ids_list[self.train_size : self.train_size + self.eval_size]), 
            np.array(attention_mask_key_list[self.train_size : self.train_size + self.eval_size])
        ]
        self.X_key_test = [
            np.array(keyphrase_ids_list[self.train_size+self.eval_size :]), 
            np.array(attention_mask_key_list[self.train_size + self.eval_size :])
        ]

        self.user_labels_train = np.array(user_labels[:self.train_size])
        self.user_labels_eval = np.array(user_labels[self.train_size : self.train_size + self.eval_size])
        self.user_labels_test = np.array(user_labels[self.train_size + self.eval_size:])

        self.user_id_list = user_id_list
        # Extract categories
        self.business_to_categories = self.get_business_to_categories()

    def resize(self, new_train_size, new_eval_size):
        """ The function is designed to resize the dataset from a particular csv file
        Goal: Gaurantee the train, eval, test size are exactly the same as the one in the combined model training.

        - This is used when we want to get the dataset for a particular user and train a single user model.
        - In combined training, the dataset size is larger (since it contains all users' reviews).
        - Hence, the split for a single user's review won't be the same as the split we only input the single user's csv.
        - However, as long as we don't shuffle the dataset when taking the single user's split,
          the single user's split can become the same as combined split, by slicing the df with the combined split size.
        """
        # Swap eval and test if size of eval is 0
        if len(self.y_eval) == 0:
            self.X_eval, self.X_test = self.X_test, self.X_eval
            self.X_key_eval, self.X_key_test = self.X_key_test, self.X_key_eval
            self.y_eval, self.y_test = self.y_test, self.y_eval
            self.user_labels_eval, self.user_labels_test = self.user_labels_test, self.user_labels_eval

        # Resize train and eval
        if new_train_size < self.train_size:
            new_X_train = self.X_train[0][:new_train_size], self.X_train[1][:new_train_size]
            new_X_key_train = self.X_key_train[0][:new_train_size], self.X_key_train[1][:new_train_size]
            new_y_train = self.y_train[:new_train_size]
            new_user_labels_train = self.user_labels_train[:new_train_size]

            new_X_eval = (
                np.concatenate([self.X_train[0][new_train_size:], self.X_eval[0]], axis=0),
                np.concatenate([self.X_train[1][new_train_size:], self.X_eval[1]], axis=0),
            )
            new_X_key_eval = (
                np.concatenate([self.X_key_train[0][new_train_size:], self.X_key_eval[0]], axis=0),
                np.concatenate([self.X_key_train[1][new_train_size:], self.X_key_eval[1]], axis=0),
            )
            new_y_eval = np.concatenate([self.y_train[new_train_size:], self.y_eval])
            new_user_labels_eval = np.concatenate([self.user_labels_train[new_train_size:], self.user_labels_eval])
        
            self.X_train = new_X_train
            self.X_key_train = new_X_key_train
            self.y_train = new_y_train
            self.user_labels_train = new_user_labels_train
        elif new_train_size > self.train_size:
            add_train_size = new_train_size - self.train_size

            new_X_train = (
                np.concatenate([self.X_train[0], self.X_eval[0][:add_train_size]], axis=0),
                np.concatenate([self.X_train[1], self.X_eval[1][:add_train_size]], axis=0),
            )
            new_X_key_train = (
                np.concatenate([self.X_key_train[0], self.X_key_eval[0][:add_train_size]], axis=0),
                np.concatenate([self.X_key_train[1], self.X_key_eval[1][:add_train_size]], axis=0),
            )
            new_y_train = np.concatenate([self.y_train, self.y_eval[:add_train_size]])
            new_user_labels_train = np.concatenate([self.user_labels_train, self.user_labels_eval[:add_train_size]])

            new_X_eval = self.X_eval[0][add_train_size:], self.X_eval[1][add_train_size:]
            new_X_key_eval = self.X_key_eval[0][add_train_size:], self.X_key_eval[1][add_train_size:]
            new_y_eval = self.y_eval[add_train_size:]
            new_user_labels_eval = self.user_labels_eval[add_train_size:]

            self.X_train = new_X_train
            self.X_key_train = new_X_key_train
            self.y_train = new_y_train
            self.user_labels_train = new_user_labels_train
        else:
            print("No change in train size")
            new_X_eval = self.X_eval
            new_X_key_eval = self.X_key_eval
            new_y_eval = self.y_eval
            new_user_labels_eval = self.user_labels_eval

        # Swap new_eval and test if size of new_eval is 0
        if len(new_y_eval) == 0:
            new_X_eval, self.X_test = self.X_test, new_X_eval
            new_X_key_eval, self.X_key_test = self.X_key_test, new_X_key_eval
            new_y_eval, self.y_test = self.y_test, new_y_eval
            new_user_labels_eval, self.user_labels_test = self.user_labels_test, new_user_labels_eval

        # Resize eval and test
        if new_eval_size < len(new_y_eval):
            self.X_eval = new_X_eval[0][:new_eval_size], new_X_eval[1][:new_eval_size]
            self.X_key_eval = new_X_key_eval[0][:new_eval_size], new_X_key_eval[1][:new_eval_size]
            self.y_eval = new_y_eval[:new_eval_size]
            self.user_labels_eval = new_user_labels_eval[:new_eval_size]

            if len(self.y_test) == 0:
                self.X_test = (
                    new_X_eval[0][new_eval_size:], new_X_eval[1][new_eval_size:]
                )
                self.X_key_test = (
                    new_X_key_eval[0][new_eval_size:], new_X_key_eval[1][new_eval_size:]
                )
                self.y_test = new_y_eval[new_eval_size:]
                self.user_labels_test = new_user_labels_eval[new_eval_size:]
            else:
                self.X_test = (
                    np.concatenate([new_X_eval[0][new_eval_size:], self.X_test[0]], axis=0),
                    np.concatenate([new_X_eval[1][new_eval_size:], self.X_test[1]], axis=0),
                )
                self.X_key_test = (
                    np.concatenate([new_X_key_eval[0][new_eval_size:], self.X_key_test[0]]),
                    np.concatenate([new_X_key_eval[1][new_eval_size:], self.X_key_test[1]]),
                )
                self.y_test = np.concatenate([new_y_eval[new_eval_size:], self.y_test])
                self.user_labels_test = np.concatenate([new_user_labels_eval[new_eval_size:], self.user_labels_test])
        elif new_eval_size > len(new_y_eval):
            add_eval_size = new_eval_size - len(new_y_eval)

            if len(self.y_test) == 0:
                # This case shouldn't occur
                raise NotImplementedError("This case shouldn't occur, the dataset is not fully resized under the case")
                self.X_eval = (
                    self.X_test[0][:add_eval_size], self.X_test[1][:add_eval_size]
                )

                self.X_key_eval = (
                    self.X_key_test[0][:add_eval_size], self.X_key_test[1][:add_eval_size]
                )

                self.y_eval = self.y_test[:add_eval_size]
                self.user_labels_eval = self.user_labels_test[:add_eval_size]
            else:
                self.X_eval = (
                    np.concatenate([new_X_eval[0], self.X_test[0][:add_eval_size]], axis=0),
                    np.concatenate([new_X_eval[1], self.X_test[1][:add_eval_size]], axis=0),
                )
                self.X_key_eval = (
                    np.concatenate([new_X_key_eval[0], self.X_key_test[0][:add_eval_size]], axis=0),
                    np.concatenate([new_X_key_eval[1], self.X_key_test[1][:add_eval_size]], axis=0),
                )
                self.y_eval = np.concatenate([new_y_eval, self.y_test[:add_eval_size]])
                self.user_labels_eval = np.concatenate([new_user_labels_eval, self.user_labels_test[:add_eval_size]])

            self.X_test = (
                self.X_test[0][add_eval_size:], self.X_test[1][add_eval_size:]
            )
            self.X_key_test = (
                self.X_key_test[0][add_eval_size:], self.X_key_test[1][add_eval_size:]
            )
            self.y_test = self.y_test[add_eval_size:]
            self.user_labels_test = self.user_labels_test[add_eval_size:]
        else:
            print("No change in eval and test size")


    def encode_data(self, model_type):
        labels = []
        input_ids_list = []
        attention_mask_list = []
        y = []
        keyphrase_ids_list = []
        attention_mask_key_list = []
        user_id_list = []
        user_labels = []
        data = self.df.values.tolist()
        # Save the slow pretrained tokenizer
        if "bert" in model_type:
            # slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            # # Load the fast tokenizer from saved file
            # save_path = "bert_base_uncased/"
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # slow_tokenizer.save_pretrained(save_path)
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        elif 'gpt2' in model_type:
            # slow_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            # save_path = "gpt2-medium/"
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # slow_tokenizer.save_pretrained(save_path)
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
        elif 'bart' in model_type:
            tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

        with tqdm(total=len(data)) as pbar:
            for v in data:
                pbar.update(1)
                item = v[0]
                review_text = v[1]
                business_name = v[2]
                name = v[3]
                k_list = literal_eval(v[4])
                user_id = v[5]
                #TODO the min review check is removed since we already check it 
                # check if enough review for the item
                # if self.item_dist[item] < self.min_reviews:
                #     continue
                # mask item name from the review
                # mask = ' '.join(['[MASK]' for x in range(len(name.split()))])

                #TODO remove all masking
                # pattern = re.compile(name, re.IGNORECASE)
                # review_text = pattern.sub(mask, review_text)
                # if self.masking:
                #     review_text = self.mask_entities(review_text)

                # Process category by removing weird characters
                categories = name.split()
                remove_list = [",", "(", ")", "/"]
                for i in range(len(categories)):
                    if categories[i] == "&":
                        continue
                    for remove in remove_list:
                        categories[i] = categories[i].replace(remove, "").lower()

                category_text = " : ".join(categories)

                # Remove duplicate keyphrase
                keyphrase_text = ""
                duplicate_key_list = []
                for keyphrase in k_list:
                    if keyphrase in duplicate_key_list:
                        continue
                    keyphrase_text = keyphrase_text + keyphrase[1] + " : "
                    duplicate_key_list.append(keyphrase)
                
                if len(k_list) > 0:
                    keyphrase_text = keyphrase_text[:-2]
                
                # Process text
                input_ids, attention_mask = self.get_input_ids_and_att_mask(
                    review_text, tokenizer, model_type
                )

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

                # Process keyphrase
                keyphrase_ids, attention_mask_key = self.get_input_ids_and_att_mask(
                    category_text + " | " + keyphrase_text, tokenizer, model_type
                )

                keyphrase_ids_list.append(keyphrase_ids)
                attention_mask_key_list.append(attention_mask_key)

                # Process categories
                # category_ids, attention_mask_cat = self.get_input_ids_and_att_mask(
                #     category_text, tokenizer, model_type
                # )

                # Process categories + keyphrase
                # cat_key_ids, attention_mask_cat_key = self.get_input_ids_and_att_mask(
                #     category_text + " | " + keyphrase_text, tokenizer, model_type
                # )
                
                # Process labels
                if item not in labels:
                    labels.append(item)
                y.append(labels.index(item))

                # Process user_id
                if user_id not in user_id_list:
                    user_id_list.append(user_id)
                user_labels.append(user_id_list.index(user_id))
        return input_ids_list, attention_mask_list, y, labels, keyphrase_ids_list, attention_mask_key_list, user_labels, user_id_list

    def get_input_ids_and_att_mask(self, text, tokenizer, model_type):
        # Process text
        tokenized_review = tokenizer.encode(text)
        input_ids = tokenized_review
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Truncate if needed
        padding_length = self.max_len - len(input_ids)
        if "bert" in model_type:
            padding_id = 0
        elif 'gpt2' in model_type:
            padding_id = 50256
        elif 'bart' in model_type:
            padding_id = 1

        if padding_length > 0:  # pad
            input_ids = input_ids + ([padding_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
        else:
            input_ids = input_ids[0:self.max_len]
            if 'bert' in model_type:
                input_ids[-1] = 102
            elif 'gpt2' in model_type:
                input_ids[-1] = 50256
            elif 'bart' in model_type:
                input_ids[-1] = 2
            attention_mask = attention_mask[0:self.max_len]
        return input_ids, attention_mask


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

if __name__ == '__main__':
    Dataset = Dataset('toronto')