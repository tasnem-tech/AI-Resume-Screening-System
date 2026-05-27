import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening System")

st.title("AI Resume Screening System")
st.write("Compare resumes with job descriptions using AI")

resume = st.text_area("Paste Resume Text")
job_desc = st.text_area("Paste Job Description")

if st.button("Calculate Match"):
    if resume and job_desc:
        text = [resume, job_desc]

        cv = CountVectorizer()
        matrix = cv.fit_transform(text)

        similarity = cosine_similarity(matrix)[0][1]
        score = round(similarity * 100, 2)

        st.subheader(f"Match Score: {score}%")

        if score > 70:
            st.success("Excellent Match")
        elif score > 40:
            st.warning("Average Match")
        else:
            st.error("Low Match")
    else:
        st.warning("Please enter both fields")
