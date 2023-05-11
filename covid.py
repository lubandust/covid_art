from geotext import GeoText
import spacy
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
import ast
import re
import gensim.downloader as api

# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_trf
nlp = spacy.load('en_core_web_sm')
tagger = SequenceTagger.load('ner')
nlp_sent = spacy.load('en_core_web_trf')
model_glove = api.load("glove-wiki-gigaword-100")

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def detect_tense(text):
    tex = text.replace('\n', ' ')
    sents = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', tex)
    tenses = {"past_tense": 0, "present_tense": 0, "future_tense": 0}
    for sent in sents:
        res = list(nlp_sent(sent).sents)[0]
        if (res.root.tag_ == "VBD") or (res.root.tag_ == "VBN"):
            tenses["past_tense"] += 1
        elif (res.root.tag_ == "VBC") or (res.root.tag_ == "VBF") or (res.root.tag_ == "VB"):
            tenses["future_tense"] += 1
        else:
            if res.root.tag_ in ["VBG", "VBP", "VBZ"]:
                tenses["present_tense"] += 1
    if tenses["future_tense"] > (len(sents) // 3):
        tense = "future_tense"
    else:
        tense = max(tenses, key=tenses.get)
    return tense

def get_number(text):
    clear_numbers = ""
    doc = nlp(text)
    for entity in doc.ents:
        if entity.label_ == 'CARDINAL' or entity.label_ == 'QUANTITY':
            clear_numbers = clear_numbers + entity.text + ', '
    return clear_numbers[:-2]

def get_gathering_amount(text):
    x = ast.literal_eval(text)
    gathering_numbers = ""
    for chunk in x:
        for word in chunk:
            word = re.sub(r'[^\w\d\s]+', '', word)
            if len(word) > 5:
                doc = nlp(word)
                if word[-1] == 's':
                    word = word[:-1]
                if ((doc[0].tag_ == 'NNP') or (doc[0].tag_ == 'NNS') or (doc[0].tag_ == 'NN')):
                    try:
                        scores = model_glove.relative_cosine_similarity("people", word)
                        non_human = model_glove.relative_cosine_similarity("object", word)
                        if ((scores >= 0.06) and (non_human <= 0.052)) or ("people" in word) or (
                                "person" in word):
                            gathering_numbers = gathering_numbers + ' '.join(
                                str(v) for v in chunk) + ', '
                        else:
                            pass
                    except KeyError:
                        pass
    return gathering_numbers

def identify_futuristic(df):
    lines = df["GATHERING_AMOUNT"].split(', ')
    lines = list(filter(None, lines))
    sents = split_into_sentences(df["TEXT"])
    numbers_sents = []
    for sent in sents:
        sent = re.sub(r'[^\w\d\s]+', '', sent)
        if any([n in sent for n in lines]):
            numbers_sents.append(sent)
    df["GATHER_SENTS"] = ', '.join(sent for sent in numbers_sents)
    tenses = {"present_tense": 0, "future_tense": 0}
    for sentence in numbers_sents:
        sentence_doc = nlp_sent(sentence)
        if any((token.morph.get('Tense') == [] and
                token.morph.get('VerbForm') == ['Fin'] and
                token.morph.get('Mood') == [])
               or
               (token.morph.get('Tense') == ['Pres'] and
                token.morph.get('VerbForm') == ['Fin'] and
                token.morph.get('Mood') != ['Ind'])
               for token in sentence_doc):
            tenses["future_tense"] += 1
        else:
            tenses["present_tense"] += 1
    tense = max(tenses, key=tenses.get)
    df['TENSE/PT/FT'] = ''.join((tense, ', ', str(tenses["present_tense"]), ', ', str(tenses["future_tense"])))
    return df

def get_geotext_articles(row):
    res = GeoText(str(row['spacy_loc']) + str(row['flair_loc']))
    mentions = []
    for city in res.cities:
        mentions.append(city)
    for country in res.countries:
        mentions.append(country)
    for nation in res.nationalities:
        mentions.append(nation)
    return mentions, res.country_mentions


def get_US_geotext_articles(row):
    res = GeoText(row['Mentions'])
    mentions = []
    for city in res.cities:
        if 'US' in GeoText(city).country_mentions:
            mentions.append(city)
    if 'US' in res.country_mentions and list(res.country_mentions.items())[0][0] == 'US':
        return True, mentions, res.country_mentions
    else:
        return False, None, None


def get_strategy_2(row, df_simplemaps):
   result_county = {}
   result_state = {}
   new_result_county = {}
   new_result_state = {}
   counties = row['counties'].split('\n')[:-1]
   states = row['states'].split('\n')[:-1]
   for location in row['US_cities']:
       result_state[location] = ''
       result_county[location] = ''
   for county in counties:
       county = county.replace("'", "")
       query = county.split(':')[0]
       result = county.split(':')[1].split(',')
       result_county[query] = result
   for state in states:
       state = state.replace("'", "")
       query = state.split(':')[0]
       result = state.split(':')[1].split(',')
       result_state[query] = result
   for location in row['US_cities']:
       if len(result_county[location]) > 1 and len(result_state[location]) > 1:
           df_county = df_simplemaps.query("city == \'" + location + "\'")
           df_county.sort_values('population')
           #print(df_county['population'].iloc[0].tolist())
           if df_county['population'].iloc[0].tolist() > 50000:
               row['strategy_2'] = row['strategy_2'] + location + ', ' + df_county['county_name'].iloc[0] + ', ' + df_county['state_name'].iloc[0] + '\n'

       if len(result_county[location]) == 1 and len(result_state[location]) > 1:
           #print(result_county[location])
           try:
               df_county = df_simplemaps.query("city == \'" + location + "\'" + " and county_name == \'" + result_county[location][0] + "\'")
               row['strategy_2'] = row['strategy_2'] + location + ', ' + result_county[location][0] + ', ' + df_county['state_name'].iloc[0] + '\n'
           except:
               pass

       if len(result_state[location]) == 1 and len(result_county[location]) > 1:
           try:
               df_county = df_simplemaps.query("city == \'" + location + "\'" + " and state_name == \'" + result_state[location][0] + "\'")
               row['strategy_2'] = row['strategy_2'] + location + ', ' + df_county['county_name'].iloc[0] + ', ' + result_state[location][0] + '\n'
           except:
               pass

       if len(result_state[location]) == 1 and len(result_county[location]) == 1:
           row['strategy_2'] = row['strategy_2'] + location + ', ' + result_county[location][0] + ', ' + result_state[location][0] + '\n'



def get_strategy_1(row, df_simplemaps):
   result_county = {}
   result_state = {}
   counties = row['counties'].split('\n')[:-1]
   states = row['states'].split('\n')[:-1]
   for location in row['US_cities']:
       result_state[location] = ''
       result_county[location] = ''
   for county in counties:
       query = county.split(':')[0]
       result = county.split(':')[1].split(',')
       if len(result) > 1:
           for county in result:
               if county in row['US_cities']:
                   result_county[query] = county
       else:
           result_county[query] = result[0]

   for state in states:
       query = state.split(':')[0]
       result = state.split(':')[1].split(',')
       if len(result) > 1:
           for state in result:
               if state in row['US_cities']:
                   result_state[query] = state
       else:
           result_state[query] = result[0]
   for location in row['US_cities']:
      if result_county[location] != '' and result_state[location] != '':
          row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'
      if result_county[location] != '' and result_state[location] == '':
          try:
              df_state = df_simplemaps.query("city == \'" + location + "\'" + " and county_name == \'" + result_county[location] + "\'")
              result_state[location] = df_state['state_name'].tolist()[0]
              row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'
          except:
              pass
      if result_county[location] == '' and result_state[location] != '':
          try:
              df_county = df_simplemaps.query("city == \'" + location + "\'" + " and state_name == \'" + result_state[location] + "\'")
              result_county[location] = df_county['county_name'].tolist()[0]
              row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'
          except:
              pass


def get_county(path, save_path):
   df_2 = pd.read_csv(path)
   df_simplemaps = pd.read_csv("uscities.csv").replace("'", "")
   co = 0
   us_co = 0
   #row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'row['strategy_1'] = row['strategy_1'] + location + ', ' + result_county[location] + ', ' + result_state[location] + '\n'
   df_2.fillna(" ",inplace=True)
   df_2.columns = ["Url", "Text", "Title", "Article_date", "spacy_loc", "flair_loc", "dates", "numbers_context", "numbers", "Mentions", "US_cities"]
   states = list(set(df_simplemaps['state_name'].tolist()))
   for index, row in df_2.iterrows():
       city_list = list(set(ast.literal_eval(str(row['US_cities']))))
       row['US_cities'] = city_list
       row['counties'] = ''
       row['states'] = ''
       row['strategy_1'] = ''
       row['strategy_2'] = ''
       if city_list:
           for city in city_list:
               if city in states:
                   row['states'] = row['states'] + city + ":" + city + '\n'
               else:
                   #print("city = \'" + city + "\'")
                   df_county = df_simplemaps.query("city == \'" + city + "\'")
                   if df_county.empty == False:
                       row['counties'] = row['counties'] + city + ":" + ','.join(df_county['county_name'].tolist())+ '\n'
                       row['states'] = row['states'] + city + ":" + ','.join(df_county['state_name'].tolist()) + '\n'
           #print(row['counties'])
           if row['counties'] != '':
               get_strategy_1(row, df_simplemaps)
               get_strategy_2(row, df_simplemaps)
               if row['strategy_1'] != '' or row['strategy_2'] != '':
                   add_to_csv(pd.DataFrame([row]), save_path)


def get_stage_1_articles(path, save_path):
   df = pd.read_csv(path)
   df.fillna("nan",inplace=True)
   df.columns = ["Url", "Text", "Title", "Article_date", "spacy_loc", "flair_loc", "dates", "numbers_context", "numbers"]
   for index, row in df.iterrows():
       if row['Text'] != 'nan':
           country_mentions, mentions_lst = get_geotext_articles(row)
           row["Mentions"] = ', '.join(country_mentions)
           if len(row["Mentions"]) > 0:
               add_to_csv(pd.DataFrame([row]), save_path)

def most_mentions(path, save_path):
   df = pd.read_csv(path)
   df.fillna("nan",inplace=True)
   df.columns = ["Url", "Text", "Title", "Article_date", "spacy_loc", "flair_loc", "dates", "numbers_context", "numbers", "Mentions"]
   for index, row in df.iterrows():
       #print(index)
       if row['Text'] != 'nan':
           #row['Text'] = remove_cookies(row['Text'])
           if_US, US_mentions, mentions_lst = get_US_geotext_articles(row)
           #print(US_mentions)
           if if_US:
               row['US_cities'] = US_mentions
               #if len(row["Mentions"]) > 0:
               add_to_csv(pd.DataFrame([row]), save_path)

def get_date(df):
    df['Date'] = df['Text'].apply(lambda x: [entity.text for entity in x.ents if entity.label_ == 'DATE'])
    return df

def get_location(df):
    possible_loc_list = []
    df['Location_spacy'] = df['Text'].apply(lambda x: [entity.text for entity in x.ents if entity.label_ == 'GPE'])
    text = [entity.text for entity in df['Text'][0] if entity.pos_ == 'PROPN']
    for i in range(len(df['Text'][0])):
        if df['Text'][0][i].pos_ == 'PROPN':
            possible_loc = df['Text'][0][i].text
            j = 1
            tag = 'PROPN'
            while tag == 'PROPN' and (i + j) < len(df['Text'][0]):
                if df['Text'][0][i + j].pos_ == 'PROPN':
                    possible_loc = possible_loc + ' ' + df['Text'][0][i + j].text
                    df['Text'][0][i + j].pos_ = 'NUM'
                    j = j + 1
                    tag = 'PROPN'
                else:
                    tag = 'NUM'
            possible_loc_list.append(possible_loc)
    new_text = ', '.join(possible_loc_list)
    locations = Sentence(new_text)
    tagger.predict(locations)
    df['Location_flair'] = str([entity.text for entity in locations.get_spans('ner') if entity.get_label("ner").value == 'LOC'])
    return df

def check_cardinal(df):
    df['Size_context'] = df['Text'].apply(lambda x: [(str(df['Text'][0][entity.start:]).split(" ", 4)[:4]) for entity in x.ents if entity.label_ == 'CARDINAL' or entity.label_ == 'QUANTITY'])
    df['Gathered_size'] = df['Text'].apply(lambda x: [entity.text for entity in x.ents if entity.label_ == 'CARDINAL' or entity.label_ == 'QUANTITY'])
    return df

def add_to_csv(df, path):
    df.to_csv(path, mode='a', header=False, index=False)

def remove_cookies(text):
    cleared_text = ""
    text_ = text.split('\n\n')
    clear = text_.copy()
    for t in clear:
        if "[" in t and "]" in t:
            text_.remove(t)
    filtr = text_.copy()
    filtr_n = []
    for i in range(len(filtr)):
        if "cookies" not in filtr[i]:
            filtr_n.append(filtr[i].strip())
    cleared_text = ' '.join(par for par in filtr_n)
    return cleared_text

def get_data_articles(path, save_path, sample):
   df = pd.read_csv(path)
   df = df.sample(n=sample)
   df.to_csv("sample.csv")
   df = pd.read_csv("sample.csv")
   df.fillna("nan",inplace=True)
   df.columns = ["idx", "Text", "Title", "Code", "Url", "Date"]
   for index, row in df.iterrows():
       if row['Text'] != 'nan':
           row['Text'] = remove_cookies(row['Text'])
           if len(row['Text']) < 300000:
               df_row = pd.DataFrame({'Url':[row['Url']]})
               df_row["Text"] = [nlp(row['Text'])]
               df_row["Title"] = row["Title"]
               df_row["Article_date"] = row["Date"]
               df_row = get_location(df_row)
               df_row = get_date(df_row)
               df_row = check_cardinal(df_row)
               df_row["Text"] = row['Text']
               add_to_csv(df_row, save_path)


def get_gather_filter(path, save_path):
    df = pd.read_csv(path)
    df.columns = ["URL", "TEXT", "TITLE", "DATE", "spacy_LOCATIONS", "flair_LOCATIONS", "DATES", "NUMBERS_CONTEXT",
                  "NUMBERS", "GEOTEXT", "GEOTEXT_UNQ", "COUNTIES", "STATES", "STRATEGY 1", "STRATEGY 2"]
    df['GATHERING_AMOUNT'] = df['NUMBERS_CONTEXT'].apply(lambda s: get_gathering_amount(s))
    df['GATHERING_NUMBER'] = df['GATHERING_AMOUNT'].apply(lambda s: get_number(s))
    df = df.apply(lambda s: identify_futuristic(s), axis=1)
    df = df[df[['GATHERING_NUMBER']].notnull().all(1)]
    df = df[~df['TENSE/PT/FT'].str.contains('future')]
    df.to_csv(save_path, index=False)
    #df.to_excel("/home/teks/PycharmProjects/trends_plots/gather_filter.xlsx")

pd.set_option('display.max_columns', None)
#sample = 1000
#get_data_articles("dataset_v5_1_full.csv", 'dataset_stage_1.csv', sample)
#print("Original rows amount: ", sample, "After removing of nan and extremely long articles: ", len(open('dataset_stage_1.csv').readlines()))
#get_stage_1_articles('dataset_stage_1.csv', 'dataset_country_mentions.csv')
#print("After removing of articles without country mentions: ", len(open('dataset_country_mentions.csv').readlines()))
#most_mentions('dataset_country_mentions.csv', 'dataset_US.csv')
#print("After removing of non-US articles: ", len(open('dataset_US.csv').readlines()))
#get_county("dataset_US.csv", "dataset_strategies.csv")
#print("After removing of articles without sufficient results in strategy 1 or 2: ", len(open('dataset_strategies.csv').readlines()))
get_gather_filter("dataset_strategies.csv", "dataset_gathering.csv")
print("After gathering filter applying: ", len(open('dataset_gathering_upd.csv').readlines()))
