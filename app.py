import streamlit as st
import pickle
from deep_translator import GoogleTranslator

model = pickle.load(open('logistic_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article below, and the model will predict whether it's **Real** or **Fake**.")


def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)


st.subheader("🧪 Try a Demo News Article")

demo_articles = {
    "— Select a demo —": {"title": "", "body": ""},
    "NASA Earth-like Planet": {
        "title": "NASA confirms Earth-like planet",
        "body": "NASA has confirmed the discovery of a new Earth-like planet located in the habitable zone of a nearby star system. Scientists believe it may support life."
    },
    "Aliens Endorse Candidate": {
        "title": "Aliens endorse presidential candidate",
        "body": "In a bizarre twist, sources claim that aliens held a secret UN meeting to endorse a presidential candidate. Officials have declined to comment."
    },
    "India GDP Growth": {
        "title": "India's GDP grows by 7.8%",
        "body": "The Ministry of Finance released data showing a 7.8% GDP growth in the first quarter of 2024, marking strong economic recovery."
    },
    "Crypto Cows in Gujarat": {
        "title": "Cows in Gujarat trade cryptocurrency",
        "body": "Reports have emerged that cows in rural Gujarat have been trained to use smartphones to participate in crypto trading. Experts remain skeptical."
    },
    "Coffee Cures Hair Loss?": {
        "title": "Coffee may cure hair loss",
        "body": "Scientists suggest that caffeine may have properties that promote hair growth, but further studies are required to confirm these findings."
    }
}

selected_demo = st.selectbox("Choose a demo input:", list(demo_articles.keys()))
demo = demo_articles[selected_demo]

st.subheader("OR try another article")

title = st.text_input("📰 Headline", value=demo['title'])
body = st.text_area("📝 Full Article Body", value=demo['body'], height=200)

if st.button("Analyze"):
    full_text = title + " " + body

    translated = translate_to_english(full_text)
    vect = vectorizer.transform([translated])

    if full_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        input_vect = vectorizer.transform([translated])

        proba = model.predict_proba(input_vect)[0]
        prediction = model.predict(input_vect)[0]

        confidence = round(max(proba) * 100, 2)

        if prediction == 1:
            st.success(f"✅ This appears to be Real News ({confidence}% confidence)")
        else:
            st.error(f"🚨 This may be Fake News ({confidence}% confidence)")