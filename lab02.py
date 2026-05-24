import nltk
import spacy
import unicodedata
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

nlp_en = spacy.load('en_core_web_sm')
nlp_pl = spacy.load('pl_core_news_sm')

def cwiczenie_2_1():
    tekst = "The researchers were studying the effectiveness of different approaches to natural language processing."
    ps = PorterStemmer()
    tokeny = word_tokenize(tekst)
    for t in tokeny:
        print(f"{t} -> {ps.stem(t)}")

def cwiczenie_2_2():
    # Odpowiedź: Lancaster jest bardziej agresywny i częściej powoduje błędy typu over-stemming, sprowadzając słowa o różnych znaczeniach do tego samego rdzenia. Porter jest bardziej umiarkowany i lepiej zachowuje różnice między tymi słowami.
    slowa = ['generous', 'generation', 'generalize', 'generally', 'generic']
    ps = PorterStemmer()
    ls = LancasterStemmer()
    for s in slowa:
        print(f"{s} | Porter: {ps.stem(s)} | Lancaster: {ls.stem(s)}")

def cwiczenie_3_1():
    # Odpowiedź: Wyniki są poprawne tylko wtedy, gdy parametr 'pos' zgadza się z faktyczną rolą słowa w zdaniu. Na przykład 'leaves' jako rzeczownik daje 'leaf', a jako czasownik 'leave'.
    slowa = ['leaves', 'produced', 'worse', 'flying', 'feet']
    wnl = WordNetLemmatizer()
    for s in slowa:
        print(f"{s} | N: {wnl.lemmatize(s, pos='n')} | V: {wnl.lemmatize(s, pos='v')}")

def cwiczenie_3_2():
    tekst = "Programiści tworzyli zaawansowane aplikacje wykorzystujące sztuczną inteligencję do analizy danych medycznych."
    doc = nlp_pl(tekst)
    lematy = [t.lemma_.lower() for t in doc if not t.is_punct]
    for t in doc:
        print(f"{t.text}\t{t.lemma_}\t{t.pos_}")
    print(f"Unikalne lematy: {len(set(lematy))}")

def cwiczenie_4_1():
    nltk_stops = set(stopwords.words('english'))
    spacy_stops = nlp_en.Defaults.stop_words
    print(f"Wspólne: {len(nltk_stops & spacy_stops)}")
    print(f"Tylko NLTK: {len(nltk_stops - spacy_stops)}")
    print(f"Tylko spaCy: {len(spacy_stops - nltk_stops)}")

def filter_with_exceptions(text, nlp, keep_words):
    doc = nlp(text)
    return [t.text for t in doc if (not t.is_stop or t.text.lower() in keep_words) and not t.is_punct]

def cwiczenie_4_3():
    # Odpowiedź: Zbiór spaCy jest bardziej agresywny, ponieważ obejmuje szerszy zakres słów uznawanych za stop words.
    tekst = "The development of artificial intelligence has changed the way we think about technology and its impact on our daily lives."
    tokens = word_tokenize(tekst.lower())
    n_stops = [t for t in tokens if t in set(stopwords.words('english'))]
    s_stops = [t for t in tokens if t in nlp_en.Defaults.stop_words]
    print(f"NLTK: {len(n_stops)/len(tokens):.3%}")
    print(f"spaCy: {len(s_stops)/len(tokens):.3%}")

def normalize_text(text, remove_numbers=False):
    text = unicodedata.normalize('NFC', text)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9ąęćłńóśźżĄĘĆŁŃÓŚŹŻ\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def normalize_social_media(text):
    text = text.replace(':)', '<POSITIVE>').replace(':(', '<NEGATIVE>').replace(':D', '<POSITIVE>')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_nltk(text, language='english', use_stemming=True):
    """Klasyczny pipeline: normalizacja → tokenizacja → stop words → stemming."""

    # 1. Normalizacja
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Tokenizacja
    tokens = word_tokenize(text, language=language)

    # 3. Usuwanie stop words
    stop_words = set(stopwords.words(language))
    tokens = [t for t in tokens if t not in stop_words]

    # 4. Stemming (opcjonalny)
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens

def preprocess_spacy(text, nlp, use_lemma=True):
    """Nowoczesny pipeline: normalizacja → spaCy (tokenizacja + POS + lematyzacja) → filtrowanie."""

    # 1. Minimalna normalizacja (spaCy radzi sobie z resztą)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Przetwarzanie przez spaCy (tokenizacja + POS + lematyzacja w jednym kroku)
    doc = nlp(text)

    # 3. Filtrowanie: bez stop words, bez interpunkcji, bez spacji
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if use_lemma:
            tokens.append(token.lemma_.lower())
        else:
            tokens.append(token.text.lower())

    return tokens

def cwiczenie_6_1():
    tekst = "Machine learning algorithms have been significantly improving over the last decade. Deep neural networks are now capable of understanding complex patterns in data, which was previously thought to be impossible."
    tokeny_nltk = preprocess_nltk(tekst)
    nlp_en = spacy.load('en_core_web_sm')
    tokeny_spacy = preprocess_spacy(tekst, nlp_en)
    print(f"NLTK - liczba tokenów: {len(tokeny_nltk)}, unikalnych: {len(set(tokeny_nltk))}")
    print(f"spaCy - liczba tokenów: {len(tokeny_spacy)}, unikalnych: {len(set(tokeny_spacy))}")

def preprocess_polish(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'https?://\S+', '', text)
    doc = nlp_pl(text)
    return [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and not t.like_num]

def cwiczenie_6_3():
    opinie = [
        "Ten film był absolutnie wspaniały! Polecam wszystkim.",
        "Nie podobał mi się ten film. Był nudny i zbyt długi.",
        "Aktorzy grali świetnie, ale fabuła mogłaby być lepsza.",
        "Najgorszy film, jaki kiedykolwiek widziałem!!!",
        "Film ok, nic specjalnego. Może warto obejrzeć na nudny wieczór.",
    ]
    for o in opinie:
        oryginalne_tokeny = [t.text for t in nlp_pl(o) if not t.is_space]
        czyste_tokeny = preprocess_polish(o)
        if len(oryginalne_tokeny) > 0:
            procent = (len(czyste_tokeny) / len(oryginalne_tokeny)) * 100
        else:
            procent = 0 
        print(f"Oryginał: {o}")
        print(f"Przetworzone: {czyste_tokeny}")
        print(f"Zachowano: {procent:.1f}% oryginalnych tokenów\n")

if __name__ == "__main__":
    #print(filter_with_exceptions("To nie jest dobre rozwiązanie", nlp_pl, ['nie', 'bardzo']))
    #print(normalize_text("W 2024 roku firma zatrudniła 150 nowych pracowników w 3 oddziałach.", remove_numbers=True))
    #print(normalize_social_media("Suuuuper film!!! :) :D Polecam!!! https://movie.com @everyone :("))
    print(preprocess_polish("W 2024 roku firma Google zaprezentowała 5 nowych modeli AI. Więcej na: https://ai.google"))
