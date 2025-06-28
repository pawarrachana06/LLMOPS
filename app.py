# app.py
import streamlit as st
from tabs import data_loading,data_preprocessing,model_training,model_evaluation,model_deployment,model_monitoring
# , training, evaluation, deployment, monitoring

st.set_page_config(page_title="T5 Summarization", layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Data Loading",
    "🧹 Preprocessing",
    "🎯 Model Training",
    "📊 Evaluation",
    "🚀 Deployment",
    "📈 Monitoring"
])

with tab1:
    data_loading.run()
with tab2:
    data_preprocessing.run()
with tab3:
    model_training.run()
with tab4:
    model_evaluation.run()
with tab5:
    model_deployment.run()
with tab6:
    model_monitoring.run()
