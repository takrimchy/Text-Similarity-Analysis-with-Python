import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


count_vect = CountVectorizer()
vectorizer = TfidfVectorizer(stop_words='english')

# Define your documents
Document1 = """Cricket is a popular sport played with a bat and ball. It originated in England and has gained immense popularity worldwide. The game is played between two teams of eleven players each, and the objective is to score more runs than the opposing team. Cricket matches are known for their intense competition and strategic gameplay."""

Document2 = """Cricket is a bat-and-ball sport that originated in England and is now played globally. It involves two teams of eleven players competing against each other. The primary aim is to score more runs than the opponent. Cricket matches are renowned for their tactical intricacies and the excitement they generate among fans."""

Document3 = """Cricket, a bat-and-ball game, originated in England and has become a widely loved sport across the globe. The game is played between two teams of eleven players each. The objective is to accumulate more runs than the opposing team. Cricket matches are known for their captivating display of skill, strategy, and excitement."""

Document4 = """Cricket, a bat-and-ball sport, has its roots in England but has gained global recognition. Played between two teams of eleven players, the game revolves around scoring more runs than the opponent. Cricket matches captivate fans with their strategic depth and thrilling moments, making it a beloved sport worldwide."""

# Remove stop words from a text
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Apply the function to your documents
Document1_no_stop = remove_stop_words(Document1)
Document2_no_stop = remove_stop_words(Document2)
Document3_no_stop = remove_stop_words(Document3)
Document4_no_stop = remove_stop_words(Document4)

corpus = [Document1_no_stop, Document2_no_stop, Document3_no_stop, Document4_no_stop]

X_train_counts = count_vect.fit_transform(corpus)
trsfm = vectorizer.fit_transform(corpus)




# Display the results
df_counts = pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names_out(), index=['Document 1', 'Document 2','Document 3','Document 4'])
df_tfidf = pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names_out(), index=['Document 1', 'Document 2','Document 3','Document 4'])
cos_sim = cosine_similarity(trsfm[0:1], trsfm)


print("Without stop words:")
print(df_counts)
print(df_tfidf)
print(cos_sim)

# Create the cosine similarity matrix
cos_sim_matrix = cosine_similarity(trsfm)

# Convert the similarity matrix to a pandas dataframe for easier viewing
similarity_df = pd.DataFrame(cos_sim_matrix, columns=range(1, len(corpus)+1), index=range(1, len(corpus)+1))

# Display the similarity matrix
sns.heatmap(similarity_df, annot=True, cmap='Blues')
plt.show()

# Create a bar chart of the cosine similarity percentages
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(1, len(corpus)+1), cos_sim_matrix[0], color='blue')
ax.set_xlabel('Document')
ax.set_ylabel('Cosine Similarity')
ax.set_xticks(range(1, len(corpus)+1))
ax.set_xticklabels(['Document {}'.format(i) for i in range(1, len(corpus)+1)])
plt.show()
