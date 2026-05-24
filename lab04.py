import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def zadanie_A():
    recenzje_ksiazek = [
        "Magiczny swiat pelen elfow i smokow, wciagajaca fabula od pierwszej strony",
        "Bohater wyrusza w epicka podroz by pokonac mrocznego wladce",
        "Swietny system magii i budowanie swiata, autor ma niesamowita wyobraznie",
        "Mroczne fantasy z moralnymi dylematami, postacie niejednoznaczne i ciekawe",
        "Piekne opisy krain i stworzen, przypomina najlepsze dziela Tolkiena",
        "Zaklecia i magiczne artefakty tworza fascynujacy element fabuly",
        "Detektyw prowadzi sledztwo w sprawie zagadkowego morderstwa w zamknietym pokoju",
        "Trzymajacy w napieciu thriller z zaskakujacym zwrotem akcji na koncu",
        "Policjant tropi seryjnego morderce po sladach zostawionych na miejscu zbrodni",
        "Zimny kryminal skandynawski pelen mrocznych tajemnic i zlozonej intrygi",
        "Sledztwo w malym miasteczku odslania mroczne sekrety mieszkancow",
        "Genialny detektyw rozwiazuje sprawe pozornie idealnej zbrodni",
    ]
    kategorie_ksiazek = ["Fantasy"]*6 + ["Kryminal"]*6

    count_vec = CountVectorizer()
    X_bow = count_vec.fit_transform(recenzje_ksiazek)
    slownik = count_vec.get_feature_names_out()
    gestosc = X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1])
    print(f"Rozmiar macierzy BoW: {X_bow.shape}")
    print(f"Rozmiar slownika: {len(slownik)}")
    print(f"Gestosc: {gestosc:.2%}")

    tfidf_vec = TfidfVectorizer()
    X_tfidf = tfidf_vec.fit_transform(recenzje_ksiazek)
    df_tfidf = pd.DataFrame(X_tfidf.toarray().round(3), columns=tfidf_vec.get_feature_names_out(), index=[f"D{i}" for i in range(len(recenzje_ksiazek))])
    
    print("\nMacierz TF-IDF:")
    print(df_tfidf.T.to_string())

    print("\nTop 3 cechy per dokument:")
    for i, r in enumerate(recenzje_ksiazek):
        wiersz = X_tfidf[i].toarray().flatten()
        top_idx = wiersz.argsort()[::-1][:3]
        cechy = [(slownik[j], round(float(wiersz[j]), 3)) for j in top_idx]
        print(f"D{i:2d} [{kategorie_ksiazek[i]}]: {cechy}")

    freq_dok = np.asarray((X_bow > 0).sum(axis=0)).flatten()
    srednia_tfidf = np.asarray(X_tfidf.mean(axis=0)).flatten()
    df_stats = pd.DataFrame({"term": slownik, "w_ilu_dok": freq_dok, "sr_tfidf": srednia_tfidf.round(4)}).sort_values("sr_tfidf", ascending=False).reset_index(drop=True)
    print("\nRanking globalny termow:")
    print(df_stats.head(10))

    cos_sim = cosine_similarity(X_tfidf)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cos_sim, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(recenzje_ksiazek)))
    ax.set_yticks(range(len(recenzje_ksiazek)))
    ax.set_xticklabels([f"D{i}" for i in range(len(recenzje_ksiazek))])
    ax.set_yticklabels([f"D{i} ({k})" for i, k in enumerate(kategorie_ksiazek)])
    
    for i in range(len(recenzje_ksiazek)):
        for j in range(len(recenzje_ksiazek)):
            ax.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", fontsize=8)
            
    plt.colorbar(im, label="Cosine similarity")
    plt.title("Podobienstwo dokumentow")
    plt.tight_layout()
    plt.show()

    konfig = [
        {"ngram_range": (1,1), "min_df": 1},
        {"ngram_range": (1,2), "min_df": 1},
        {"ngram_range": (1,1), "min_df": 2},
        {"ngram_range": (1,2), "min_df": 2},
    ]
    print("\nLiczba cech przy roznych ngram_range i min_df:")
    for cfg in konfig:
        vec = TfidfVectorizer(**cfg)
        X = vec.fit_transform(recenzje_ksiazek)
        sparsity = 1 - X.nnz / (X.shape[0] * X.shape[1])
        print(f"Konfiguracja {cfg} -> Cechy: {X.shape[1]}, Rzadkosc: {sparsity:.4f}")

def zadanie_B():
    posty = [
        "Trening interwalowy to najlepszy sposob na poprawe kondycji biegowej",
        "Reprezentacja zdobyla zloty medal na mistrzostwach swiata w siatkowce",
        "Nowy rekord swiata w maratonie pobity o trzy sekundy",
        "Trener oglosil powolania na zgrupowanie kadry przed meczem",
        "Dieta i regeneracja sa rownie wazne jak sam trening sportowy",
        "Premiera nowego smartfona z aparatem o rozdzielczosci stu megapikseli",
        "Sztuczna inteligencja zmienia sposob w jaki pracujemy i uczymy sie",
        "Aktualizacja systemu operacyjnego przynosi nowe funkcje bezpieczenstwa",
        "Robot wykorzystujacy sztuczna inteligencje pomaga w diagnostyce medycznej",
        "Nowy laptop z procesorem najnowszej generacji i ekranem OLED",
        "Domowy chleb na zakwasie wymaga cierpliwosci ale smakuje wysmienicie",
        "Przepis na szarlotke z kruszonka i cynamonem na jesienne wieczory",
        "Kuchnia azjatycka laczy ostre przyprawy z delikatnymi sosami",
        "Pieczony kurczak z ziolami prowansalskimi i pieczonymi warzywami",
        "Sezon na grzyby to idealny czas na domowy krem z borowikow",
    ]
    kategorie_postow = ["Sport"]*5 + ["Tech"]*5 + ["Kuchnia"]*5

    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
    X_tfidf = tfidf_vec.fit_transform(posty)
    
    vec_uni = TfidfVectorizer(ngram_range=(1, 1))
    X_uni = vec_uni.fit_transform(posty)
    
    print(f"\nRozmiar macierzy same unigramy: {X_uni.shape}")
    print(f"Rozmiar macierzy unigramy+bigramy: {X_tfidf.shape}")

    slownik = tfidf_vec.get_feature_names_out()
    print("\nTop 5 cech per dokument:")
    for i, p in enumerate(posty):
        wiersz = X_tfidf[i].toarray().flatten()
        top_idx = wiersz.argsort()[::-1][:5]
        cechy = [(slownik[j], round(float(wiersz[j]), 3)) for j in top_idx]
        print(f"D{i:2d}: {cechy}")

    def znajdz_podobne(query, n=3):
        q_vec = tfidf_vec.transform([query])
        sim = cosine_similarity(q_vec, X_tfidf).flatten()
        top_idx = sim.argsort()[::-1][:n]
        print(f"\nZapytanie: '{query}'")
        for idx in top_idx:
            print(f"  -> {posty[idx]} (Podobienstwo: {sim[idx]:.3f}, Kategoria: {kategorie_postow[idx]})")

    znajdz_podobne("trening silowy i odzywanie sportowcow")
    znajdz_podobne("nowy smartfon z aparatem i sztuczna inteligencja")
    znajdz_podobne("przepis na domowe ciasto drozdzowe")

if __name__ == "__main__":
    zadanie_A()
    zadanie_B()