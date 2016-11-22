import json

def load_article_data(article_file_path):
    with open(article_file_path,encoding='utf8') as fd:
        articles = fd.readlines()
        articles = list(filter(lambda a: a != '\n', articles))
    return articles

def get_sentences(path,article):
    total_path = path + article
    scoreLines = load_article_data(total_path)
    return scoreLines
    
def get_sentence_corefs(indx,article):
    #find the length of simple,stand documents
    simple = 117
    stand = 8
    #get the index of the simple and the stand sentence corresponding to the indx given
    simple_indx = indx/simple
    stand_indx = indx % stand
    
    #get the corefs for simple sentence
    json_data = open('../data/'+article+'_simple_coref_json').read()
    new_json = json.loads(json_data)
    simple_corefs = new_json[simple_indx + 1]
    
    stand_corefs = new_json[stand_indx + 1]
    
    return simple_corefs,stand_corefs
    
    
    
    