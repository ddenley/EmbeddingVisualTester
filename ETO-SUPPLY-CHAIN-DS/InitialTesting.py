import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np


csv_file = "CSV_FILES\inputs.csv"

#Process CSV file
def csv_to_df(csv_file):
    df = pd.read_csv(csv_file)
    df = df[['input_name', 'type', 'stage_name', 'description']]
    return df

#Inspect CSV file
def inspect_df(df):
    print(df.head())
    print(df.info())
    print(df.describe())

def generate_embeddings(df):
    #model = SentenceTransformer('bert-base-nli-mean-tokens')
    model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
    embeddings = model.encode(df['description'], show_progress_bar=True)
    return embeddings


def type_to_color(df):
    #Get unique types
    types = df['type'].unique()
    #Generate unique colors for each type
    colormap = plt.cm.get_cmap('hsv', len(types))
    #Create dictionary of types and colors
    type_color_dict = {doc_type: colormap(i)[:3] for i, doc_type in enumerate(types)}
    return type_color_dict

def create_color_array(df, type_color_dict):
    colors = []
    for index, row in df.iterrows():
        colors.append(type_color_dict[row['type']])
    return colors


def generate_umap(embeddings, dimensions):
    reducer = umap.UMAP(n_neighbors=15, n_components=dimensions, metric='cosine')
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

#myavi 3d visualization
def visualize_umap_3d(umap_embeddings):
    colors = create_color_array(df, type_to_color(df))
    x = umap_embeddings[:, 0]
    y = umap_embeddings[:, 1]
    z = umap_embeddings[:, 2]
    for i in range(len(x)):
        mlab.points3d(x[i], y[i], z[i], color=colors[i], scale_factor=0.1)
        mlab.text3d(x[i], y[i], z[i] + 0.1, df['input_name'].iloc[i], scale=0.05, color=(0, 0, 0))
    mlab.show()
    wait = input()



if __name__ == '__main__':
    df = csv_to_df(csv_file)
    print(type_to_color(df))
    embeddings = generate_embeddings(df)
    umap_embeddings = generate_umap(embeddings, 3)
    visualize_umap_3d(umap_embeddings)