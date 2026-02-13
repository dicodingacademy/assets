import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- FUNGSI VISUALISASI ---
def ai_vs_productivity(df):
    fig = plt.figure(figsize=(8,6))
    df['ai_usage_group'] = pd.cut(
        df['ai_tool_usage_hours_per_week'],
        bins=[0, 2, 5, 10, 20],
        labels=['0–2 jam', '3–5 jam', '6–10 jam', '>10 jam'],
        include_lowest=True
    )
    sns.barplot(data=df, x='ai_usage_group', y='productivity_score', errorbar=None)
    plt.title('Rata-rata Produktivitas berdasarkan Intensitas Penggunaan AI')
    plt.xlabel('Penggunaan AI per Minggu')
    plt.ylabel('Rata-rata Skor Produktivitas')
    st.pyplot(fig)

def error_rate_vs_automation(df):
    fig = plt.figure(figsize=(8,6))
    df['automation_group'] = pd.cut(
        df['tasks_automated_percent'],
        bins=[0, 25, 50, 75, 100],
        labels=['0–25%', '26–50%', '51–75%', '76–100%'],
        include_lowest=True
    )
    sns.barplot(data=df, x='automation_group', y='error_rate_percent', errorbar=None)
    plt.title('Rata-rata Error Rate berdasarkan Tingkat Otomasi')
    plt.xlabel('Persentase Tugas yang Diotomasi')
    plt.ylabel('Rata-rata Error Rate (%)')
    st.pyplot(fig)

def ai_vs_job_role(df):
    fig = plt.figure(figsize=(10,6))
    order = df.groupby('job_role')['productivity_score'].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x='job_role', y='productivity_score', order=order, errorbar=None)
    plt.title('Efektivitas AI terhadap Produktivitas di Tiap Peran Pekerjaan')
    plt.xlabel('Peran Pekerjaan')
    plt.ylabel('Rata-rata Skor Produktivitas')
    plt.xticks(rotation=30)
    st.pyplot(fig)

def ai_usage_by_job_role(df):
    fig = plt.figure(figsize=(10,6))
    order = df.groupby('job_role')['ai_tool_usage_hours_per_week'].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x='job_role', y='ai_tool_usage_hours_per_week', order=order, errorbar=None)
    plt.title('Intensitas Penggunaan AI pada Setiap Peran Pekerjaan')
    plt.xlabel('Peran Pekerjaan')
    plt.ylabel('Rata-rata Jam Penggunaan AI per Minggu')
    plt.xticks(rotation=30)
    st.pyplot(fig)

def correlation_burnout_factors(df):
    fig = plt.figure(figsize=(12,7))
    corr_vals = df.corr(numeric_only=True)['burnout_risk_score'].drop('burnout_risk_score').sort_values()
    corr_vals.plot(kind='barh', color='salmon')
    plt.title('Seberapa Kuat Faktor Mempengaruhi Burnout')
    plt.xlabel('Nilai Korelasi')
    st.pyplot(fig)

def meeting_hours_vs_productivity(df):
    fig = plt.figure(figsize=(12, 6))
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    labels = ['0–5 jam', '6–10 jam', '11–15 jam', '16–20 jam', '21–25 jam', '26–30 jam', '31–35 jam', '36–40 jam']
    df['meeting_range'] = pd.cut(df['meeting_hours_per_week'], bins=bins, labels=labels, include_lowest=True)
    sns.lineplot(data=df, x='meeting_range', y='productivity_score', marker='o', errorbar=None)
    plt.title('Dampak Jam Rapat terhadap Penurunan Produktivitas')
    plt.xlabel('Total Jam Rapat per Minggu')
    plt.ylabel('Rata-rata Skor Produktivitas')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def experience_vs_productivity(df):
    fig = plt.figure(figsize=(12, 8))
    df['exp_group'] = pd.cut(df['experience_years'], bins=[0,2,5,10,20], labels=['0-2 th','3-5 th','6-10 th','>10 th'])
    sns.barplot(data=df, x='exp_group', y='productivity_score', errorbar=None)
    plt.title('Rata-rata Produktivitas berdasarkan Pengalaman Kerja')
    plt.xlabel('Kelompok Pengalaman')
    plt.ylabel('Rata-rata Produktivitas')
    st.pyplot(fig)

def profile_top_performers(df):
    fig = plt.figure(figsize=(12, 8))
    threshold = df['productivity_score'].quantile(0.9)
    df['performance_category'] = np.where(df['productivity_score'] >= threshold, 'Top 10% Performer', 'Others')
    features_to_compare = ['experience_years', 'ai_tool_usage_hours_per_week', 'tasks_automated_percent', 'focus_hours_per_day']
    profile_comparison = df.groupby('performance_category')[features_to_compare].mean().T
    profile_plot_data = profile_comparison.reset_index().melt(id_vars='index')
    profile_plot_data.columns = ['Feature', 'Category', 'Average Value']
    sns.barplot(data=profile_plot_data, x='Average Value', y='Feature', hue='Category', palette='viridis')
    plt.title('Profil Karyawan: Top 10% Produktivitas vs Lainnya', fontsize=15)
    st.pyplot(fig)

# Konfigurasi Halaman
st.set_page_config(page_title="AI Productivity Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('df.csv')
    return df

df = load_data()

# --- LOAD MODEL & ENCODERS ---
@st.cache_resource
def load_ml_components():
    data = joblib.load('rf_model.joblib')
    return data['model'], data['encoders'], data['features']

try:
    rf_model, label_encoders, features_used = load_ml_components()
except FileNotFoundError:
    st.error("File 'rf_model.pkl' tidak ditemukan. Pastikan kamu sudah menyimpan model dan encoder dengan pickle.")

# --- INITIALIZE SESSION STATE ---
# Ini untuk menyimpan halaman yang sedang aktif agar tidak hilang saat klik tombol lain
if 'page' not in st.session_state:
    st.session_state.page = "📊 Visualisasi Data"

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="https://www.dicoding.com/blog/wp-content/uploads/2014/12/dicoding-header-logo.png" width="200">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### Menu")
    
    # Tombol pertama: Visualisasi
    if st.button("📊 Visualisasi Data", use_container_width=True, 
                 type="primary" if st.session_state.page == "📊 Visualisasi Data" else "secondary"):
        st.session_state.page = "📊 Visualisasi Data"
        st.rerun()
    
    # Tombol kedua: Prediksi
    if st.button("🔍 Prediksi Data", use_container_width=True,
                 type="primary" if st.session_state.page == "🔍 Prediksi Data" else "secondary"):
        st.session_state.page = "🔍 Prediksi Data"
        st.rerun()

    st.markdown("---")
    st.info("💡 **Tips:** Gunakan halaman prediksi untuk simulasi risiko Burnout pada karyawan.")

# --- RENDER HALAMAN BERDASARKAN PILIHAN SIDEBAR ---

if st.session_state.page == "📊 Visualisasi Data":
    st.title("AI Workplace Productivity Dashboard")
    st.markdown("Dashboard ini menampilkan tren data terkait penggunaan AI di tempat kerja.")
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sejauh mana penggunaan AI benar-benar meningkatkan produktivitas?")
        ai_vs_productivity(df)
    with col2:
        st.markdown("### Apakah tingginya tingkat otomasi berbanding lurus dengan pengurangan kesalahan atau error?")
        error_rate_vs_automation(df)

    st.markdown("### Peran pekerjaan mana yang paling efektif dalam memanfaatkan AI? dan bagaimana intensitas penggunaannya?")
    col3, col4 = st.columns(2)
    with col3:
        ai_vs_job_role(df)
    with col4:
        ai_usage_by_job_role(df)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("### Apa pemicu utama risiko burnout di perusahaan?")
        st.markdown("")
        st.markdown("")
        correlation_burnout_factors(df)
    with col6:
        st.markdown("### Berapa batas ideal jam rapat sebelum produktivitas mulai menurun tajam?")
        meeting_hours_vs_productivity(df)

    col7, col8 = st.columns(2)
    with col7:
        st.markdown("### Apakah tingkat pengalaman kerja menjamin produktivitas yang lebih tinggi?")
        experience_vs_productivity(df)
    with col8:
        st.markdown("### Bagaimana profil karyawan dengan produktifitas terbaik?")
        st.markdown("")
        st.markdown("")
        profile_top_performers(df)

elif st.session_state.page == "🔍 Prediksi Data":
    st.title("Prediksi Risiko Burnout Karyawan")
    st.markdown("Masukkan data karyawan untuk memprediksi tingkat risiko burnout secara real-time.")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            job_role = st.selectbox("Peran Pekerjaan", label_encoders['job_role'].classes_)
            exp = st.number_input("Tahun Pengalaman", 0, 40, 5)
            ai_hours = st.number_input("Jam Pakai AI/Minggu", 0.0, 50.0, 10.0)
            learn_hours = st.number_input("Jam Belajar/Minggu", 0.0, 20.0, 2.0)
            
        with c2:
            auto_pct = st.slider("% Tugas Otomatis", 0.0, 100.0, 30.0)
            manual_hours = st.slider("Jam Kerja Manual/Minggu", 0.0, 50.0, 20.0)
            pressure = st.selectbox("Tekanan Deadline", label_encoders['deadline_pressure_level'].classes_)
            collab_hours = st.number_input("Jam Kolaborasi/Minggu", 0.0, 30.0, 5.0)
            
        with c3:
            meeting_hours = st.slider("Jam Rapat/Minggu", 0.0, 50.0, 10.0)
            focus_hours = st.slider("Jam Fokus/Hari", 1.0, 12.0, 5.0)
            wlb_score = st.slider("Skor Work-Life Balance (1-10)", 1.0, 10.0, 7.0)
            error_rate = st.number_input("Tingkat Kesalahan (%)", 0.0, 10.0, 1.0)
            
        submit = st.form_submit_button("Prediksi Risiko Sekarang")
    
    if submit:
        complexity = 50.0 
        prod_score = 70.0

        input_data = pd.DataFrame({
            'job_role': [job_role],
            'experience_years': [exp],
            'ai_tool_usage_hours_per_week': [ai_hours],
            'tasks_automated_percent': [auto_pct],
            'manual_work_hours_per_week': [manual_hours],
            'learning_time_hours_per_week': [learn_hours],
            'deadline_pressure_level': [pressure],
            'meeting_hours_per_week': [meeting_hours],
            'collaboration_hours_per_week': [collab_hours],
            'error_rate_percent': [error_rate],
            'task_complexity_score': [complexity],
            'focus_hours_per_day': [focus_hours],
            'work_life_balance_score': [wlb_score],
            'productivity_score': [prod_score]
        })

        input_data['burnout_risk_score'] = 0.0  # Nilai dummy

        input_data = input_data[features_used]

        for col in ['job_role', 'deadline_pressure_level']:
            input_data[col] = label_encoders[col].transform(input_data[col])
        
        prediction_code = rf_model.predict(input_data)
        prediction_label = label_encoders['burnout_risk_level'].inverse_transform(prediction_code)[0]

        st.subheader(f"Hasil Prediksi: {prediction_label}")
        if prediction_label == 'High':
            st.error("Risiko Burnout Tinggi! Perlu penyesuaian beban kerja.")
        elif prediction_label == 'Medium':
            st.warning("Risiko Burnout Sedang. Pantau jam fokus karyawan.")
        else:
            st.success("Risiko Burnout Rendah. Pertahankan ritme kerja!")