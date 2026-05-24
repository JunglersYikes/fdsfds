import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset


sns.set_theme(style="whitegrid")

label_map = {
    "__label__meta_plus_m": "pozytywny",
    "__label__meta_minus_m": "negatywny",
    "__label__meta_zero": "neutralny",
    "__label__meta_amb": "ambiwalentny"
}

dataset_in = load_dataset("allegro/klej-polemo2-in")
df_in = dataset_in["train"].to_pandas()
df_in["label"] = df_in["target"].map(label_map)
df_in["num_words"] = df_in["sentence"].str.split().str.len()

dataset_out = load_dataset("allegro/klej-polemo2-out")
df_out = dataset_out["train"].to_pandas()
df_out["label"] = df_out["target"].map(label_map)

print("--- Podstawowe statystyki ---")
print(f"Liczba przykładów (train): {df_out.shape[0]}")
print(f"Liczba unikalnych etykiet: {df_out['target'].nunique()}")
print(f"Występowanie klas:\n{df_out['label'].value_counts()}")
print(f"\nProporcje klas:\n{df_out['label'].value_counts(normalize=True).round(3)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_in["label"].value_counts().plot(kind="bar", ax=axes[0], color="#3498db", edgecolor="black")
axes[0].set_title("Rozkład klas In-Domain")
axes[0].set_xlabel("Sentyment")
axes[0].tick_params(axis="x", rotation=0)

df_out["label"].value_counts().plot(kind="bar", ax=axes[1], color="#e74c3c", edgecolor="black")
axes[1].set_title("Rozkład klas Out-of-Domain")
axes[1].set_xlabel("Sentyment")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()

df_out["num_chars"] = df_out["sentence"].str.len()
df_out["num_words"] = df_out["sentence"].str.split().str.len()

print("\n--- Średnia i mediana długości w podziale na klasy ---")
print(df_out.groupby("label")["num_words"].agg(["mean", "median"]))

plt.figure(figsize=(10, 6))
plt.hist(
    [df_in["num_words"], df_out["num_words"]], 
    bins=30, 
    label=["In-Domain", "Out-of-Domain"], 
    color=["#3498db", "#e74c3c"], 
    alpha=0.7
)
plt.title("Porównanie rozkładu długości tekstów (liczba słów)")
plt.xlabel("Liczba słów")
plt.ylabel("Liczba recenzji")
plt.legend()
plt.show()

idx_max = df_out["num_words"].idxmax()
idx_min = df_out["num_words"].idxmin()

print("\n--- Najdłuższa recenzja ---")
print(f"Etykieta: {df_out.loc[idx_max, 'label']}")
print(f"Treść: {df_out.loc[idx_max, 'sentence']}\n")

print("--- Najkrótsza recenzja ---")
print(f"Etykieta: {df_out.loc[idx_min, 'label']}")
print(f"Treść: {df_out.loc[idx_min, 'sentence']}")