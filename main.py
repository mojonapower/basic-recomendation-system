import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#leer datos y transformarlos en DataFrame
df = pd.read_csv("actividades.csv")
#quita espacios innecesarios
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()
#todos los tags los ordena en un arreglo por fila
df = df.melt(id_vars=['nombre actividad', 'nivel'], value_vars=df.filter(regex='tag').columns, value_name='tag')
df = df.dropna().groupby(['nombre actividad', 'nivel'])['tag'].apply(list).reset_index()

# Crear una matriz de características para cada actividad
activities = df["nombre actividad"].tolist()
all_tags = set([tag for tags in df["tag"] for tag in tags])

#########################


feature_matrix = []
for index, row in df.iterrows():
    features = [0] * len(all_tags)
    for tag in row["tag"]:
        features[list(all_tags).index(tag)] = 1
    feature_matrix.append(features)

# Calcular la similitud entre las actividades
similarity = cosine_similarity(feature_matrix)

# Función para recomendar actividades similares
def recommend_activities(activity, df, similarity, top_n=5):
    index = df.index[df["nombre actividad"] == activity][0]
    sim = similarity[index]
    top_indices = sim.argsort()[-top_n:][::-1]
    recommended_activities = [df.iloc[i]["nombre actividad"] for i in top_indices]
    return recommended_activities

# Recomendar actividades para una actividad dada
activity = "Funciones en JavaScript"
recommended = recommend_activities(activity, df, similarity)
print("Recommended activities for '{}': {}".format(activity, recommended))