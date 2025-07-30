import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Updated data
data = {
    'User': ['Sai', 'Sai', 'Teja', 'Teja', 'Kiran', 'Kiran'],
    'Movie': ['Bahubali', 'Pushpa', 'Bahubali', 'Aarya', 'Pushpa', 'Aarya'],
    'Rating': [5, 4, 5, 3, 4, 5]
}

df = pd.DataFrame(data)

# Create user-movie matrix
table = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Calculate similarity between users
similarity = cosine_similarity(table)
sim_df = pd.DataFrame(similarity, index=table.index, columns=table.index)

# Pick a user and find most similar
target = 'Sai'
match = sim_df[target].drop(target).idxmax()

# Recommend movies watched by match but not by target
target_movies = set(df[df['User'] == target]['Movie'])
match_movies = df[df['User'] == match]

recommend = match_movies[~match_movies['Movie'].isin(target_movies)]

print(f"Movies recommended for {target} based on {match}'s ratings:")
print(recommend[['Movie', 'Rating']])
