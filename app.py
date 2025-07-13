import streamlit as st
import pickle
import os

# Try importing deep_translator, fall back to no translation if unavailable
try:
    from deep_translator import GoogleTranslator

    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    st.warning("Translation functionality is not available. Please ensure 'deep-translator' is installed.")


# Load models with proper error handling
@st.cache_resource
def load_models():
    try:
        # Load the logistic regression model
        with open('logistic_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.error("Please ensure both 'logistic_model.pkl' and 'tfidf_vectorizer.pkl' are in the repository")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


# Load the models
model, vectorizer = load_models()

# Set page configuration
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article below, and the model will predict whether it's **Real** or **Fake**.")


def translate_to_english(text):
    """Translate text to English if translation is available"""
    if TRANSLATION_AVAILABLE:
        try:
            translator = GoogleTranslator(source='auto', target='en')
            return translator.translate(text)
        except Exception as e:
            st.warning(f"Translation failed: {e}. Using original text.")
            return text
    else:
        return text


# Demo articles section
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

# Input fields
title = st.text_input("📰 Headline", value=demo['title'])
body = st.text_area("📝 Full Article Body", value=demo['body'], height=200)

# Analysis button
if st.button("Analyze", type="primary"):
    if title.strip() == "" and body.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Combine title and body
            full_text = f"{title.strip()} {body.strip()}".strip()

            # Translate if needed
            translated_text = translate_to_english(full_text)

            # Show translation info if text was translated
            if TRANSLATION_AVAILABLE and translated_text != full_text:
                st.info("🌐 Text was translated to English for analysis.")

            try:
                # Vectorize the text
                text_vectorized = vectorizer.transform([translated_text])

                # Make prediction
                prediction = model.predict(text_vectorized)[0]
                prediction_proba = model.predict_proba(text_vectorized)[0]

                # Calculate confidence
                confidence = round(max(prediction_proba) * 100, 2)

                # Display results
                st.subheader("📊 Analysis Results")

                if prediction == 1:
                    st.success(f"✅ **Real News** ({confidence}% confidence)")
                    st.balloons()
                else:
                    st.error(f"🚨 **Fake News** ({confidence}% confidence)")

                # Show probability breakdown
                with st.expander("📈 Detailed Probability Breakdown"):
                    fake_prob = prediction_proba[0] * 100
                    real_prob = prediction_proba[1] * 100

                    st.write(f"**Fake News Probability:** {fake_prob:.2f}%")
                    st.write(f"**Real News Probability:** {real_prob:.2f}%")

                    # Create a simple progress bar visualization
                    st.write("**Confidence Distribution:**")
                    st.progress(real_prob / 100)
                    st.write(f"Real: {real_prob:.1f}% | Fake: {fake_prob:.1f}%")

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")

# Footer
st.markdown("---")
st.markdown(
    "*This tool uses machine learning to detect potentially fake news. Results should be used as a guide and not as definitive truth.*")