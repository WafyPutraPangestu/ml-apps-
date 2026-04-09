import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🎯 Ticket Priority ML Dashboard")
st.markdown("---")

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🚀 Status", "Ready")

with col2:
    st.metric("📊 Prediksi", "Coming Soon")

with col3:
    st.metric("🎯 Akurasi", "~89%")

st.markdown("---")

# Sample Chart
st.subheader("📊 Distribusi Prioritas (Sample)")

sample_data = pd.DataFrame({
    'Prioritas': ['Rendah', 'Sedang', 'Tinggi'],
    'Jumlah': [300, 400, 300]
})

col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
        sample_data,
        values='Jumlah',
        names='Prioritas',
        color='Prioritas',
        color_discrete_map={
            'Rendah': '#00CC96',
            'Sedang': '#FFA15A',
            'Tinggi': '#EF553B'
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        sample_data,
        x='Prioritas',
        y='Jumlah',
        color='Prioritas',
        color_discrete_map={
            'Rendah': '#00CC96',
            'Sedang': '#FFA15A',
            'Tinggi': '#EF553B'
        }
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.info("💡 Dashboard akan menampilkan data real setelah API menerima request")

st.subheader("📡 API Information")
st.code("""
POST http://localhost:5001/predict

Headers:
  X-Api-Key: rahasia-super-aman-123
  Content-Type: application/json

Body:
{
  "judul": "Internet mati",
  "deskripsi": "Sudah 3 jam",
  "kategori_gangguan": "Gangguan Jaringan",
  "kategori_pelanggan": "Perusahaan"
}
""")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")