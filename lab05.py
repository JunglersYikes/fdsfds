import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader as api

def zadanie_A():
    print("Ładowanie modelu GloVe (może zająć chwilę)...")
    model = api.load("glove-wiki-gigaword-100")
    
    print("\n--- Krok 1: Wektor słowa 'university' ---")
    vec_uni = model['university']
    print(f"Kształt wektora: {vec_uni.shape}")
    print(f"Pierwsze 10 wartości: {vec_uni[:10]}")
    
    print("\n--- Krok 2: 10 najbliższych sąsiadów ---")
    slowa_testowe = ["science", "music", "football"]
    for slowo in slowa_testowe:
        print(f"Najbliżsi sąsiedzi dla '{slowo}':")
        for w, score in model.most_similar(slowo, topn=10):
            print(f"  {w:15s} {score:.4f}")
            
    print("\n--- Krok 3: Podobieństwo cosinusowe między parami ---")
    pary = [("doctor", "nurse"), ("doctor", "airplane"), ("happy", "sad"), ("happy", "joyful")]
    for w1, w2 in pary:
        sim = model.similarity(w1, w2)
        print(f"Podobieństwo {w1} - {w2}: {sim:.4f}")
        
    print("\n--- Krok 4: doesnt_match ---")
    grupy_testowe = [
        ["apple", "banana", "orange", "car"],
        ["dog", "cat", "mouse", "table"]
    ]
    for grupa in grupy_testowe:
        outlier = model.doesnt_match(grupa)
        print(f"W grupie {grupa} nie pasuje: '{outlier}'")
        
    print("\n--- Krok 5: Analogie wektorowe ---")
    analogie = [
        (["japan", "paris"], ["tokyo"], "japan - tokyo + paris"),
        (["teacher", "hospital"], ["school"], "teacher - school + hospital"),
        (["slow", "faster"], ["slower"], "slow - slower + faster")
    ]
    for pos, neg, label in analogie:
        res = model.most_similar(positive=pos, negative=neg, topn=1)
        print(f"{label} = {res[0][0]} ({res[0][1]:.4f})")
        
    print("\n--- Krok 6: Wizualizacja t-SNE dla 4 grup slow ---")
    grupy = {
        "sporty": ["football", "tennis", "basketball", "swimming", "volleyball"],
        "zawody": ["doctor", "teacher", "engineer", "lawyer", "nurse"],
        "jedzenie": ["pizza", "pasta", "bread", "cheese", "rice"],
        "emocje": ["happy", "sad", "angry", "scared", "surprised"]
    }
    
    words, vectors, labels = [], [], []
    for label, group in grupy.items():
        for w in group:
            words.append(w)
            vectors.append(model[w])
            labels.append(label)
            
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    coords = tsne.fit_transform(np.array(vectors))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_map = {"sporty": "red", "zawody": "blue", "jedzenie": "green", "emocje": "purple"}
    plotted_labels = set()
    
    for i, w in enumerate(words):
        group = labels[i]
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=colors_map[group], s=60,
                   label=group if group not in plotted_labels else "")
        plotted_labels.add(group)
        ax.annotate(w, (coords[i, 0]+1, coords[i, 1]+1), fontsize=9)
        
    ax.legend()
    ax.set_title("Wizualizacja t-SNE dla 4 grup słów")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    zadanie_A()