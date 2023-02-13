import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Crear un dataframe de ejemplo con actividades y tags
data = pd.DataFrame({
    "activity": ["Activity 1", "Activity 2", "Activity 3", "Activity 4", "Activity 5"],
    "tags": [["Tag 1", "Tag 2", "Tag 3"],
            ["Tag 2", "Tag 4"],
            ["Tag 1", "Tag 3", "Tag 5"],
            ["Tag 1", "Tag 2"],
            ["Tag 3", "Tag 4", "Tag 5"]]
})

# Crear una matriz de características para cada actividad
activities = data["activity"].tolist()
all_tags = set([tag for tags in data["tags"] for tag in tags])

feature_matrix = []
for index, row in data.iterrows():
    features = [0] * len(all_tags)
    for tag in row["tags"]:
        features[list(all_tags).index(tag)] = 1
    feature_matrix.append(features)

# Calcular la similitud entre las actividades
similarity = cosine_similarity(feature_matrix)

# Función para recomendar actividades similares
def recommend_activities(activity, data, similarity, top_n=3):
    index = data.index[data["activity"] == activity][0]
    sim = similarity[index]
    top_indices = sim.argsort()[-top_n:][::-1]
    recommended_activities = [data.iloc[i]["activity"] for i in top_indices]
    return recommended_activities

# Recomendar actividades para una actividad dada
activity = "Activity 1"
recommended = recommend_activities(activity, data, similarity)
print("Recommended activities for '{}': {}".format(activity, recommended))
