"""
Tweet Sentiment Analysis System
BE Computer Engineering | Information Retrieval (IR) | SPPU 2019 Pattern

Pipeline (explain this to ma'am):
  1. Dataset      -> tweets with labels (Positive / Negative / Neutral)
  2. Preprocessing-> lowercase, remove URLs/@mentions/#, remove stopwords
  3. Features     -> TF-IDF (Term Frequency - Inverse Document Frequency)
  4. Classifier   -> Logistic Regression
  5. Evaluation   -> Accuracy, Precision, Recall, F1, Confusion Matrix
  6. Application  -> Streamlit web app (single tweet + bulk CSV)

Run:  streamlit run app.py
"""

import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------
# 1. BUILT-IN DATASET (so the app works without any download)
# ---------------------------------------------------------------
POSITIVE = [
    "I love this phone, the camera is amazing",
    "What a beautiful day, feeling so happy",
    "Just got my dream job, so excited",
    "Best movie I have seen this year, loved it",
    "Thank you so much, you made my day",
    "The food was delicious and the staff were so kind",
    "Great service, will definitely come back",
    "Proud of my team, we did an awesome job",
    "This new song is fantastic, on repeat all day",
    "Loving the weather today, perfect for a walk",
    "Congratulations to my best friend, so happy for you",
    "The concert was incredible, what a night",
    "Really enjoyed the book, highly recommend it",
    "Wonderful trip with family, great memories",
    "My exam went really well, feeling great",
    "This app is so easy to use, brilliant work",
    "Excellent customer support, problem solved quickly",
    "Super excited for the weekend, cannot wait",
    "Such a lovely surprise, I am so grateful",
    "Won the match today, what a thrilling game",
    "I love my new laptop, it is super fast",
    "Happy birthday to my lovely sister, enjoy your day",
    "This is the best day ever, so happy",
    "Amazing performance by the team, I love it",
    "Good morning everyone, have a wonderful day",
    "The new update is great, everything works smoothly",
    "So proud and happy, thank you all",
    "Loved the new restaurant, the pizza was amazing",
    "What a fantastic lecture, I learned so much",
    "Good vibes only, life is beautiful",
]
NEGATIVE = [
    "I hate this phone, the battery dies so fast",
    "Worst service ever, totally disappointed",
    "Stuck in traffic for two hours, so annoying",
    "The food was cold and the staff were rude",
    "This movie was terrible, a complete waste of time",
    "Feeling really sad and tired today",
    "Very angry, my order arrived broken again",
    "Awful internet, nothing works properly",
    "I failed my exam, feeling horrible",
    "Never buying from this shop again, so bad",
    "The app keeps crashing, absolutely useless",
    "Bad weather ruined our trip, so frustrating",
    "Terrible experience, the staff ignored us",
    "I am so bored and upset with everything",
    "This update is garbage, it broke everything",
    "Poor quality product, I want my money back",
    "Lost my wallet today, worst day ever",
    "The match was boring and the team played badly",
    "Disgusting smell in the restaurant, never again",
    "Customer care is useless, nobody replied to me",
    "I hate my new laptop, it is so slow",
    "This is the worst day ever, so sad",
    "Not good at all, very disappointed with the service",
    "The new update is bad, nothing works now",
    "Horrible lecture, I did not learn anything",
    "I am very angry and upset, this is not fair",
    "Terrible battery and a bad camera, do not buy",
    "Not happy with this product, it is a total waste",
    "Feeling lonely and sad tonight",
    "The pizza was cold and tasted awful, never again",
]
NEUTRAL = [
    "The meeting is scheduled for 3 pm tomorrow",
    "I am going to the market to buy vegetables",
    "The train departs from platform number four",
    "Today is Tuesday and the office opens at nine",
    "Just reached home after college",
    "The new phone model will launch next month",
    "Reading a chapter on information retrieval",
    "Submitting the assignment before the deadline",
    "The library is located near the main gate",
    "Having tea and checking my notes",
    "The bus route has changed since Monday",
    "Our class starts at ten in the morning",
    "Watching the news about the election results",
    "The report has twenty pages and five charts",
    "Going to the gym after lunch",
    "Weather forecast says it may rain in the evening",
    "The package was delivered to the front desk",
    "Team meeting about the project is on Friday",
    "Downloading the dataset for my project",
    "The store closes at eight on weekdays",
    "The new laptop has 8 GB RAM and a 512 GB drive",
    "Birthday party is on Saturday at the community hall",
    "The lecture is on information retrieval in room 204",
    "Good morning, the class starts in ten minutes",
    "The update will be installed tonight at midnight",
    "The restaurant opens at noon and closes at ten",
    "Leaving for college now, will reach by nine",
    "The exam timetable has been uploaded on the website",
    "Collected my notes from the library today",
    "The match will be played at the city stadium on Sunday",
]


def builtin_dataset():
    rows = (
        [(t, "Positive") for t in POSITIVE]
        + [(t, "Negative") for t in NEGATIVE]
        + [(t, "Neutral") for t in NEUTRAL]
    )
    return pd.DataFrame(rows, columns=["text", "label"])


def load_uploaded(file):
    """Accepts (a) a CSV with 'text' and 'label' columns, or
    (b) the Sentiment140 CSV from Kaggle (6 columns, no header)."""
    df = pd.read_csv(file, encoding="latin-1")
    df.columns = [str(c).lower() for c in df.columns]
    if "text" in df.columns and "label" in df.columns:
        out = df[["text", "label"]]
    else:
        file.seek(0)
        raw = pd.read_csv(file, encoding="latin-1", header=None)
        if raw.shape[1] != 6:
            raise ValueError("CSV needs 'text' and 'label' columns, or be Sentiment140 format.")
        out = raw[[5, 0]].copy()
        out.columns = ["text", "label"]
        out["label"] = out["label"].map({0: "Negative", 2: "Neutral", 4: "Positive"})
    out = out.dropna()
    if len(out) > 20000:  # keep training fast
        out = out.sample(20000, random_state=42)
    return out


# ---------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------
# Keep negation words ("not", "never") because they flip sentiment
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "never"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove links
    text = re.sub(r"@\w+", " ", text)              # remove @mentions
    text = text.replace("#", " ")                  # keep hashtag word
    text = re.sub(r"[^a-z\s]", " ", text)          # remove numbers/punctuation/emoji
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


# ---------------------------------------------------------------
# 3 + 4 + 5. TF-IDF  ->  Logistic Regression  ->  Evaluation
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Training model...")
def train_model(df):
    df = df.copy()
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 1)),    # single words (unigrams)
        LogisticRegression(C=10, max_iter=1000),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = list(model.classes_)
    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T.round(2)
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=[f"Actual {l}" for l in labels],
        columns=[f"Pred {l}" for l in labels],
    )
    # Small datasets give a noisy single split, so also report 5-fold cross-validation
    cv_acc = None
    if len(df) <= 5000:
        cv_acc = cross_val_score(model, df["clean"], df["label"], cv=5).mean()
    return model, accuracy_score(y_test, y_pred), cv_acc, report, cm, len(X_train), len(X_test)


def predict_one(model, text):
    probs = model.predict_proba([clean_text(text)])[0]
    scores = dict(zip(model.classes_, probs))
    best = max(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------
# 6. STREAMLIT APP
# ---------------------------------------------------------------
EMOJI = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}


def main():
    st.set_page_config(page_title="Tweet Sentiment Analysis", page_icon="🐦", layout="centered")
    st.title("🐦 Tweet Sentiment Analysis System")
    st.caption("Information Retrieval Mini Project | TF-IDF + Logistic Regression")

    # ----- Sidebar: choose dataset -----
    st.sidebar.header("Dataset")
    source = st.sidebar.radio("Train on", ["Built-in sample (90 tweets)", "Upload my own CSV"])
    if source == "Upload my own CSV":
        file = st.sidebar.file_uploader("CSV: 'text','label' columns or Sentiment140", type="csv")
        if file is None:
            st.info("Upload a CSV in the sidebar, or switch to the built-in sample.")
            st.stop()
        try:
            data = load_uploaded(file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    else:
        data = builtin_dataset()

    model, acc, cv_acc, report, cm, n_train, n_test = train_model(data)
    st.sidebar.success(f"Model ready | Accuracy: {(cv_acc or acc):.0%}")

    tab1, tab2, tab3 = st.tabs(["✍️ Analyze Tweet", "📂 Bulk CSV", "📊 Model Performance"])

    # ----- Tab 1: single tweet -----
    with tab1:
        tweet = st.text_area("Enter a tweet", placeholder="I love this new phone, the camera is amazing!")
        if st.button("Analyze", type="primary"):
            if not tweet.strip():
                st.warning("Please type a tweet first.")
            else:
                label, scores = predict_one(model, tweet)
                st.subheader(f"{EMOJI.get(label, '')} {label}  ({scores[label]:.0%} confidence)")
                st.bar_chart(pd.Series(scores, name="Probability"))
                with st.expander("See preprocessing step"):
                    st.write("**Original:**", tweet)
                    st.write("**After cleaning:**", clean_text(tweet))

    # ----- Tab 2: bulk CSV -----
    with tab2:
        st.write("Upload a CSV that has a **text** column. Every tweet gets a sentiment.")
        bulk = st.file_uploader("Tweets CSV", type="csv", key="bulk")
        if bulk is not None:
            df = pd.read_csv(bulk, encoding="latin-1")
            df.columns = [str(c).lower() for c in df.columns]
            if "text" not in df.columns:
                st.error("No 'text' column found.")
            else:
                df = df.dropna(subset=["text"]).head(5000)
                df["sentiment"] = model.predict(df["text"].apply(clean_text))
                st.dataframe(df)
                st.bar_chart(df["sentiment"].value_counts())
                st.download_button("Download results", df.to_csv(index=False), "results.csv", "text/csv")

    # ----- Tab 3: performance -----
    with tab3:
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{acc:.0%}")
        c2.metric("Cross-val accuracy (5-fold)", f"{cv_acc:.0%}" if cv_acc else "n/a")
        st.caption(f"Trained on {n_train} tweets, tested on {n_test}. "
                   "Small data = noisy score; upload Sentiment140 for a reliable one.")
        st.write("**Precision / Recall / F1**")
        st.dataframe(report)
        st.write("**Confusion Matrix**")
        st.dataframe(cm)
        st.write("**Most influential words**")
        vec = model.named_steps["tfidfvectorizer"]
        clf = model.named_steps["logisticregression"]
        names = vec.get_feature_names_out()
        for i, cls in enumerate(clf.classes_):
            top = names[clf.coef_[i].argsort()[-8:][::-1]]
            st.write(f"{EMOJI.get(cls, '')} **{cls}:** " + ", ".join(top))


if __name__ == "__main__":
    main()
