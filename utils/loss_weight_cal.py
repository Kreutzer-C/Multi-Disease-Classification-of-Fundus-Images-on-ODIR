import pandas as pd

df = pd.read_csv("./dataset/ODIR-5K_Annotations.csv")

label_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
total = len(df)
pos_weight = []
for label in label_names:
    positives = int(df.loc[:, [label]].sum().iloc[0])
    negatives = total - positives
    weight = negatives / positives
    pos_weight.append(weight)