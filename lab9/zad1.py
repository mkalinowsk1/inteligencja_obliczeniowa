import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os

with open('article.txt', 'r', encoding='utf-8') as f:
	text = f.read()

text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)

tokens = word_tokenize(text)
print(f"Liczba słów po tokenizacji: {len(tokens)}")

stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if not w in stop_words]
print(f"Liczba słów po usunięciu stop words: {len(filtered_tokens)}")

manual_stopwords = ['2026', 'mclaren']
stop_words.update(manual_stopwords)

again_filtered_tokens = [w for w in filtered_tokens if not w in manual_stopwords]
print(f"Liczba słów po dodaniu ręcznych stop-words: {len(again_filtered_tokens)}")

lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(w) for w in again_filtered_tokens]
print(f"e) Wybrany lematyzer: WordNetLemmatizer")
print(f"e) Liczba słów po lematyzacji: {len(lemmas)}")

final_words = lemmas

word_counts = Counter(final_words)
print("\nf) Wektor zliczający słowa (fragment):")
for word, count in list(word_counts.items())[:5]:
    print(f"  {word}: {count}")

top_10 = word_counts.most_common(10)
words, counts = zip(*top_10)

plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.xlabel('Słowa (X)')
plt.ylabel('Liczba wystąpień (Y)')
plt.title('10 najczęściej występujących słów (Bag of Words)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

text_for_cloud = " ".join(final_words)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_cloud)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Chmura Tagów (Word Cloud)')
plt.show()