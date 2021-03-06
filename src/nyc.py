import pandas as pd
import numpy as np

nyc_zones = [153, 128, 127, 243, 120, 244, 116, 42, 152, 166, 
    41, 74, 24, 43, 75, 151, 238, 239, 143, 142, 236, 263, 262, 237,
    141, 140, 229, 162, 163, 161, 230, 48, 50, 246, 68, 100, 164,
    170, 233, 137, 186, 90, 234, 107, 224, 4, 79, 113, 114,
    249, 158, 125, 211, 144, 148, 232, 45, 231, 13, 261, 12, 88, 87, 209]
nyc_pairs = [(153, 127), (153, 128), (127, 128), (128, 243), (127, 243),
    (120, 243), (120, 127), (244, 243), (244, 120), (116, 244), 
    (42, 116), (42, 120), (152, 116), (152, 42), (166, 152), (41, 166), (41, 42),
    (74, 42), (74, 41), (24, 166), (43, 41), (43, 24), (75, 74), (75, 43),
    (151, 24), (151, 43), (238, 151), (238, 43), (239, 238), (239, 43),
    (143, 239), (142, 143), (142, 239), (142, 43), (236, 75), (236, 43),
    (263, 75), (263, 236), (262, 263), (237, 43), (237, 236),
    (141, 237), (141, 263), (140, 141), (140, 262), (229, 140), (229, 141),
    (162, 229), (162, 237), (163, 162), (163, 237), (163, 43), (161, 162),
    (161, 163), (230, 163), (230, 161), (48, 230), (48, 163), (48, 142),
    (50, 143), (50, 48), (246, 50), (246, 48), (68, 246), (68, 48),
    (100, 48), (100, 68), (100, 230), (164, 100), (164, 161), (170, 164), (170, 162), (170, 161), 
    (233, 229), (233, 162), (233, 170), (137, 233), (137, 170), (186, 164), (186, 100), (186, 68),
    (90, 68), (90, 186), (90, 234), (234, 164), (107, 234), (107, 170), (107, 137),
    (224, 107), (224, 137), (4, 224), (79, 4), (79, 224), (79, 107), (113, 234), (113, 79),
    (114, 113), (114, 79), (249, 113), (249, 114), (249, 90), (249, 68),
    (158, 249), (158, 68), (125, 158), (125, 249), (211, 125), (211, 114),
    (144, 211), (144, 114), (148, 144), (148, 79), (232, 148), (232, 4),
    (45, 232), (45, 148), (45, 144), (231, 125), (231, 211), (231, 144), (231, 45),
    (13, 231), (13, 261), (231, 261), (12, 261), (12, 13), (12, 88), (88, 261),
    (88, 87), (87, 261), (209, 87), (209, 231), (209, 45)]

nyc_N = len(nyc_zones)
nyc_zones_to_vertices = {nyc_zones[i]: i for i in range(nyc_N)}
nyc_vertices_to_zones = {i: nyc_zones[i] for i in range(nyc_N)}
nyc_vertex_pairs = [(nyc_zones_to_vertices[a], nyc_zones_to_vertices[b])
                    for a, b in nyc_pairs]

nyc_queries_raw = pd.read_csv('data/nyc-queries.csv', index_col=0).values
zones_to_vertices_vect = np.vectorize(lambda s: nyc_zones_to_vertices[s])
nyc_queries = zones_to_vertices_vect(nyc_queries_raw)

def raw_csv_to_queries(path):
    df = pd.read_csv(path)
    df_nona = df[['PULocationID', 'DOLocationID']].dropna()
    df_f = df_nona[(df_nona['PULocationID'].apply(lambda n: n in nyc_zones_to_vertices)) &
               (df_nona['DOLocationID'].apply(lambda n: n in nyc_zones_to_vertices))]
    return df_f.reset_index().drop(columns=['index'])

#dfl = pd.read_csv('data/yellow_tripdata_2020-01.csv')
#dfl.head(100000).to_csv('data/yellow_tripdata_2020-01-small.csv')
#df = raw_csv_to_queries('data/yellow_tripdata_2020-01.csv')
#df.to_csv('data/nyc-queries.csv')
#df.head(100000).to_csv('data/nyc-queries-small.csv')

