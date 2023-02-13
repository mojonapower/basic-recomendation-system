import pandas as pd

#leer datos y transformarlos en DataFrame
df = pd.read_csv("actividades.csv")
#quita espacios innecesarios
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()
#todos los tags los ordena en un arreglo por fila
df = df.melt(id_vars=['nombre actividad', 'nivel'], value_vars=df.filter(regex='tag').columns, value_name='tag')
df = df.dropna().groupby(['nombre actividad', 'nivel'])['tag'].apply(list).reset_index()

print(df)