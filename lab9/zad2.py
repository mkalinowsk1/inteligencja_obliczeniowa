import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
import text2emotion as te

positive_review = "Hotel is set in an Amazing position and very convenient. Very comfortable bed and pillows, bathroom is very adequate."
negative_review = "Room was not ready and we had to wait for 2 hours. They did not clean our room four days"

vader = SentimentIntensityAnalyzer()

def analyze_vader(text, name):
    vs = vader.polarity_scores(text)
    print(f"\n--- Analiza: {name} ---")
    print(f"  Negatywny (neg): {vs['neg']:.4f}")
    print(f"  Neutralny (neu): {vs['neu']:.4f}")
    print(f"  Pozytywny (pos): {vs['pos']:.4f}")
    print(f"  Wynik zagregowany (compound): {vs['compound']:.4f}")

analyze_vader(positive_review, "Opinia Pozytywna")
analyze_vader(negative_review, "Opinia Negatywna")

print("\n## C) NARZĘDZIE 1: TextBlob")

def analyze_textblob(text, name):
    analysis = TextBlob(text)
    # Polarity: -1 (neg) do 1 (pos); Subjectivity: 0 (obiektywny) do 1 (subiektywny)
    print(f"\n--- Analiza TextBlob: {name} ---")
    print(f"  Polarity (Sentyment): {analysis.sentiment.polarity:.4f}")
    print(f"  Subjectivity (Subiektywność): {analysis.sentiment.subjectivity:.4f}")
    sentiment_tag = "Pozytywny" if analysis.sentiment.polarity > 0.1 else ("Negatywny" if analysis.sentiment.polarity < -0.1 else "Neutralny")
    print(f"  Oznaczenie: {sentiment_tag}")

analyze_textblob(positive_review, "Opinia Pozytywna")
analyze_textblob(negative_review, "Opinia Negatywna")


print("\n## C) NARZĘDZIE 2: NRCLex (Analiza Emocji)")

def analyze_nrclex(text, name):
    emotions = NRCLex(text)
    # Wyciąganie ogólnego sentymentu i top 5 emocji
    print(f"\n--- Analiza NRCLex: {name} ---")
    print(f"  Ogólny sentyment: {emotions.affect_dict.get('sentiment')}")
    print("  Emocje (tagowane):")
    # Filtrowanie i sortowanie emocji
    emotion_scores = {k: v for k, v in emotions.raw_emotion_scores.items() if v > 0}
    top_emotions = dict(sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)[:5])
    for emotion, score in top_emotions.items():
        print(f"    {emotion.capitalize()}: {score}")

analyze_nrclex(positive_review, "Opinia Pozytywna")
analyze_nrclex(negative_review, "Opinia Negatywna")


print("\n## C) NARZĘDZIE 3: Text2Emotion")
print("\n--- Analiza Text2Emotion: Opinia Pozytywna ---")
print(te.get_emotion(positive_review))
print("\n--- Analiza Text2Emotion: Opinia Negatywna ---")
print(te.get_emotion(negative_review))

#pip install emoji==1.7.0
