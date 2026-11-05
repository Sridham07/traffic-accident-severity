import pandas as pd

chunks = pd.read_csv("data/US_Accidents_March23.csv", chunksize=500_000)
df = pd.concat(c.sample(frac=0.065, random_state=42) for c in chunks)
df.to_csv("data/us_accidents_sample.csv", index=False)
print(df.shape)