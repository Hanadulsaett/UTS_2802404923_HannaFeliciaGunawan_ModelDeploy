import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Career Analytics", layout="wide")

model_cls = joblib.load('model_cls.pkl')
model_reg = joblib.load('model_reg.pkl')

# side bar
st.sidebar.header("Identitas Mahasiswa")
st.sidebar.image("https://png.pngtree.com/element_our/20190528/ourmid/pngtree-the-graduation-season-ushered-in-our-graduation--figure-2-image_1148438.jpg")
st.sidebar.write("**Nama:** Hanna Felicia Gunawan")
st.sidebar.write("**NIM:** 2802404923")
st.sidebar.divider()
st.sidebar.info("Harap isi form di halaman utama untuk melakukan prediksi.")

# main page
st.title("🎓 Student Career Analytics")
st.markdown("Silakan masukkan data mahasiswa di bawah ini untuk melihat hasil analisis.")

with st.expander("Form Input Data Mahasiswa", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["ECE", "IT", "ME", "CSE", "EE"])
        cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
        tenth = st.slider("10th Percentage", 0.0, 100.0, 75.0)
        twelfth = st.slider("12th Percentage", 0.0, 100.0, 75.0)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])

    with col_b:
        study_hours = st.number_input("Study Hours/Day", 0.0, 24.0, 5.0)
        attendance = st.slider("Attendance %", 0.0, 100.0, 85.0)
        projects = st.number_input("Projects Completed", 0, 20, 2)
        internships = st.number_input("Internships", 0, 10, 1)
        coding = st.slider("Coding Skill", 1, 5, 3)
        comm = st.slider("Communication Skill", 1, 5, 3)
        tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    with col_c:
        aptitude = st.slider("Aptitude Skill", 1, 5, 3)
        hackathons = st.number_input("Hackathons", 0, 10, 1)
        certs = st.number_input("Certifications", 0, 10, 1)
        sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
        stress = st.slider("Stress Level", 1, 10, 5)
        part_time = st.selectbox("Part Time Job", ["Yes", "No"])
        internet = st.selectbox("Internet Access", ["Yes", "No"])
        extra = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

input_data = pd.DataFrame([{
    'gender': gender, 'branch': branch, 'cgpa': cgpa, 'tenth_percentage': tenth,
    'twelfth_percentage': twelfth, 'backlogs': backlogs, 'study_hours_per_day': study_hours,
    'attendance_percentage': attendance, 'projects_completed': projects,
    'internships_completed': internships, 'coding_skill_rating': coding,
    'communication_skill_rating': comm, 'aptitude_skill_rating': aptitude,
    'hackathons_participated': hackathons, 'certifications_count': certs,
    'sleep_hours': sleep, 'stress_level': stress, 'part_time_job': part_time,
    'family_income_level': income, 'city_tier': tier, 'internet_access': internet,
    'extracurricular_involvement': extra
}])

st.divider()

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.subheader("Prediction: Placement")
    if st.button("Predict Placement Status", use_container_width=True):
        prediction = model_cls.predict(input_data)
        if prediction[0] == "Placed":
            st.success("### HASIL: **PLACED** ")
        else:
            st.error("### HASIL: **NOT PLACED** ")

with res_col2:
    st.subheader("Prediction: Salary")
    if st.button("Estimate Annual Salary", use_container_width=True):
        salary = model_reg.predict(input_data)
        st.warning(f"### ESTIMASI: **{salary[0]:.2f} LPA**")
