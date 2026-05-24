import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab', quiet=True)

def cwiczenie_3_1():
    tekst = "I've been working at OpenAI since Jan. 2020, and it's been great!"
    tokeny = word_tokenize(tekst)
    print(f"3.1 Liczba tokenów: {len(tokeny)}")

def cwiczenie_3_2(tekst):
    tokeny = word_tokenize(tekst)
    for i, token in enumerate(tokeny, start=1):
        print(f"{i}: {token}")

def cwiczenie_3_3():
    tekst = "Mrs. O'Brien can't believe it's 3:45 p.m. already!"
    print(f"3.3 Split: {len(tekst.split())}")
    print(f"3.3 NLTK: {len(word_tokenize(tekst))}")

def cwiczenie_4_1():
    tekst = "The experiment failed... Again. The results (see Fig. 2) were inconclusive. We need more data."
    zdania = sent_tokenize(tekst)
    print(f"4.1 Liczba zdań: {len(zdania)}")

def cwiczenie_4_2(tekst):
    zdania = sent_tokenize(tekst)
    tokeny = word_tokenize(tekst)
    liczba_zdan = len(zdania)
    liczba_tokenow = len(tokeny)
    srednia = liczba_tokenow / liczba_zdan if liczba_zdan > 0 else 0
    print(f"Zdań: {liczba_zdan}, Tokenów: {liczba_tokenow}, Średnia: {srednia:.2f}")

def cwiczenie_4_3():
    tekst = "Partially due to the popularity of Genshin Impact, the game was widely anticipated before its launch. It was received generally positively by critics, who praised its story, characters and combat system. However, opinions were mixed on its character progression system and gacha monetization strategy."
    zdania = sent_tokenize(tekst)
    for i, zdanie in enumerate(zdania, start=1):
        print(f"Zdanie {i}: {word_tokenize(zdanie)}")

def cwiczenie_5_1():
    tekst = "Warszawa jest stolica Polski. Liczy ok. 1,8 mln mieszkancow. W miescie znajduje sie wiele zabytkow, m.in. Zamek Krolewski i Lazienki."
    zdania = sent_tokenize(tekst, language='polish')
    for i, zdanie in enumerate(zdania, start=1):
        tokeny = word_tokenize(zdanie, language='polish')
        print(f"PL Zdanie {i}: {tokeny}")

def cwiczenie_5_2():
    tekst = "Warszawa jest stolica Polski. Liczy ok. 1,8 mln mieszkancow. W miescie znajduje sie wiele zabytkow, m.in. Zamek Krolewski i Lazienki."
    print(f"PL: {word_tokenize(tekst, language='polish')}")
    print(f"EN: {word_tokenize(tekst)}")

def count_unique_tokens(text, language):
    tokeny = word_tokenize(text, language=language)
    print(f"Unikalne: {len(set(tokeny))}")

if __name__ == "__main__":
    cwiczenie_5_2()