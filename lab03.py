import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
from wordcloud import WordCloud

nltk.download('punkt_tab', quiet=True)
stop_pl = {"i", "w", "na", "z", "nie", "to", "się", "jest", "a", "do", "że", "o", "tak", "ale", "po", "za", "od", "jeśli"}

def zadanie1():
    df = pd.read_csv("recenzje_filmowe.csv")
    print("--- Zadanie 1 ---")
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    return df

def zadanie2(df):
    df["tokens"] = df["text"].apply(lambda x: word_tokenize(str(x), language="polish"))
    df["n_tokens"] = df["tokens"].apply(len)
    wszystkie_tokeny = [t for lista in df["tokens"] for t in lista]
    print("\n--- Zadanie 2 ---")
    print(f"Dokumenty: {len(df)}")
    print(f"Tokeny: {len(wszystkie_tokeny)}")
    print(f"Typy: {len(set(wszystkie_tokeny))}")
    print(f"Średnia: {df['n_tokens'].mean():.2f}")
    print(f"Mediana: {df['n_tokens'].median()}")
    print(f"Min/Max: {df['n_tokens'].min()}/{df['n_tokens'].max()}")
    return df, wszystkie_tokeny

def zadanie3(df):
    print("\n--- Zadanie 3 ---")
    plt.hist(df["n_tokens"], bins=15, edgecolor="black")
    plt.axvline(df["n_tokens"].median(), color="red", linestyle="--", label=f"Mediana: {df['n_tokens'].median():.0f}")
    plt.xlabel("Liczba tokenów")
    plt.ylabel("Liczba dokumentów")
    plt.title("Rozkład długości dokumentów")
    plt.tight_layout()
    plt.legend()
    plt.show()

def zadanie4(wszystkie_tokeny):
    tokeny_slow = [t.lower() for t in wszystkie_tokeny if t.isalpha() and len(t) > 1]
    ttr = len(set(tokeny_slow)) / len(tokeny_slow) if tokeny_slow else 0
    print("\n--- Zadanie 4 ---")
    print(f"TTR: {ttr:.4f}")
    return tokeny_slow

def zadanie5(tokeny_slow):
    czestosc = Counter(tokeny_slow)
    hapaxy = [w for w, c in czestosc.items() if c == 1]
    print("\n--- Zadanie 5 ---")
    print(f"Hapaxy: {len(hapaxy)}")
    print(f"Procent słownika: {(len(hapaxy)/len(czestosc)*100):.2f}%")
    print(f"Przykłady: {hapaxy[:5]}")

def zadanie6(tokeny_slow):
    czestosc = Counter(tokeny_slow)
    print("\n--- Zadanie 6 ---")
    print("Top 20 ze stop words:", czestosc.most_common(20))
    bez_stop = Counter([t for t in tokeny_slow if t not in stop_pl])
    print("Top 20 bez stop words:", bez_stop.most_common(20))
    return bez_stop

def zadanie7(df):
    df["ttr"] = df["tokens"].apply(lambda t: len(set(t)) / len(t) if len(t) > 0 else 0)
    df["avg_word_len"] = df["tokens"].apply(lambda t: np.mean([len(w) for w in t]) if t else 0)
    df["stop_ratio"] = df["tokens"].apply(lambda t: len([w for w in t if w.lower() in stop_pl]) / len(t) if t else 0)
    print("\n--- Zadanie 7 ---")
    print(df[["n_tokens", "ttr", "avg_word_len", "stop_ratio"]].describe())

def zadanie8(tokeny_slow):
    tekst = " ".join([t for t in tokeny_slow if t not in stop_pl])
    wc = WordCloud(width=800, height=400, background_color="white").generate(tekst)
    plt.imshow(wc)
    plt.title("Chmura słów")
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def zadanie9(bez_stop):
    slowa, licznik = zip(*bez_stop.most_common(25))
    plt.barh(slowa, licznik)
    plt.xlabel("Częstość")
    plt.title("25 najczęstszych słów (bez stop words)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def zadanie10(tokeny_slow):
    bigramy = list(ngrams(tokeny_slow, 2))
    print("\n--- Zadanie 10 ---")
    print("Bigramy:", Counter(bigramy).most_common(15))
    filtry = [(a, b) for a, b in bigramy if a not in stop_pl and b not in stop_pl]
    print("Bigramy bez stop words:", Counter(filtry).most_common(15))

def zadanie11(tokeny_slow):
    czestosc = Counter(tokeny_slow)
    wszystkie_tokeny = sum(czestosc.values())
    pokrycie_100 = sum(c for _, c in czestosc.most_common(100)) / wszystkie_tokeny
    print(f"Top 100 słów pokrywa {pokrycie_100*100:.2f}% tokenów korpusu")
    rangi = range(1, len(czestosc) + 1)
    wartosci_czestosci = sorted(czestosc.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.loglog(rangi, wartosci_czestosci, label="Dane z korpusu")
    C = wartosci_czestosci[0]
    zipf_teoretyczny = [C / r for r in rangi]
    plt.loglog(rangi, zipf_teoretyczny, "--", color="red", alpha=0.7, label="Zipf teoretyczny")
    plt.xlabel("Ranga (log)")
    plt.ylabel("Częstość (log)")
    plt.title("Prawo Zipfa — wykres log-log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

if __name__ == "__main__":
    df_wynik = zadanie1()
    df_wynik, tokeny_wynik = zadanie2(df_wynik)
    zadanie3(df_wynik)
    slowa_wynik = zadanie4(tokeny_wynik)
    zadanie5(slowa_wynik)
    czestosc_czysta = zadanie6(slowa_wynik)
    zadanie7(df_wynik)
    zadanie8(slowa_wynik)
    zadanie9(czestosc_czysta)
    zadanie10(slowa_wynik)
    zadanie11(slowa_wynik)