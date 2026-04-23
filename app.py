import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Career Analytics", layout="wide")

model_cls = joblib.load('model_cls.pkl')
model_reg = joblib.load('model_reg.pkl')

st.title("Student Career Analytics")
st.markdown("Aplikasi Prediksi placement status dan Estimasi Gaji Mahasiswa")

st.sidebar.header("Input Data Mahasiswa")

def get_user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    branch = st.sidebar.selectbox("Branch", ["ECE", "IT", "ME", "CSE", "EE"])
    cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0, 8.0)
    tenth = st.sidebar.slider("10th Percentage", 0.0, 100.0, 75.0)
    twelfth = st.sidebar.slider("12th Percentage", 0.0, 100.0, 75.0)
    backlogs = st.sidebar.number_input("Backlogs", 0, 10, 0)
    study_hours = st.sidebar.number_input("Study Hours per Day", 0.0, 24.0, 5.0)
    attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 85.0)
    projects = st.sidebar.number_input("Projects Completed", 0, 20, 2)
    internships = st.sidebar.number_input("Internships Completed", 0, 10, 1)
    coding = st.sidebar.slider("Coding Skill Rating", 1, 5, 3)
    comm = st.sidebar.slider("Communication Skill Rating", 1, 5, 3)
    aptitude = st.sidebar.slider("Aptitude Skill Rating", 1, 5, 3)
    hackathons = st.sidebar.number_input("Hackathons Participated", 0, 10, 1)
    certs = st.sidebar.number_input("Certifications Count", 0, 10, 1)
    sleep = st.sidebar.number_input("Sleep Hours", 0.0, 12.0, 7.0)
    stress = st.sidebar.slider("Stress Level", 1, 10, 5)
    part_time = st.sidebar.selectbox("Part Time Job", ["Yes", "No"])
    income = st.sidebar.selectbox("Family Income Level", ["Low", "Medium", "High"])
    tier = st.sidebar.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
    extra = st.sidebar.selectbox("Extracurricular", ["Low", "Medium", "High"])

    data = {
        'gender': gender, 'branch': branch, 'cgpa': cgpa, 'tenth_percentage': tenth,
        'twelfth_percentage': twelfth, 'backlogs': backlogs, 'study_hours_per_day': study_hours,
        'attendance_percentage': attendance, 'projects_completed': projects,
        'internships_completed': internships, 'coding_skill_rating': coding,
        'communication_skill_rating': comm, 'aptitude_skill_rating': aptitude,
        'hackathons_participated': hackathons, 'certifications_count': certs,
        'sleep_hours': sleep, 'stress_level': stress, 'part_time_job': part_time,
        'family_income_level': income, 'city_tier': tier, 'internet_access': internet,
        'extracurricular_involvement': extra
    }
    return pd.DataFrame([data])

input_df = get_user_input()

st.subheader("Data Mahasiswa yang Diinput")
st.write(input_df)

col1, col2 = st.columns(2)

with col1:
    st.info("### Prediksi Placement Status")
    if st.button("Cek Placement Status"):
        prediction = model_cls.predict(input_df)
        if prediction[0] == "Placed":
            st.success("Hasil: **PLACED**")
        else:
            st.error("Hasil: **NOT PLACED**")

with col2:
    st.info("### Estimasi Gaji")
    if st.button("Hitung Estimasi Gaji"):
        salary = model_reg.predict(input_df)
        st.success(f"Estimasi: **{salary[0]:.2f} LPA**")

st.divider()
st.caption("2802404923 - Hanna Felicia Gunawan.")