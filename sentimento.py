import streamlit as st
import csv
import os
from datetime import datetime
from pysentimiento import create_analyzer

# Configurações
CSV_FILE = "resultados.csv"
POS_THRESHOLD = 0.85
NEG_THRESHOLD = 0.85

@st.cache_resource
def obter_analisador():
    return create_analyzer(task="sentiment", lang="pt")

# Função para salvar no CSV
def salvar_resultado(timestamp, texto, label, score):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "texto", "label", "score"])
        writer.writerow([timestamp, texto, label, f"{score:.2f}"])

# Função de análise
def analisar_texto(analisador, texto):
    resultado = analisador.predict(texto)
    label = resultado.output
    score = resultado.probas[label]
    if label == "POS" and score >= POS_THRESHOLD:
        interpret = "😀 Este texto parece amigável!"
    elif label == "NEG" and score >= NEG_THRESHOLD:
        interpret = "😠 Este texto parece bravo ou negativo."
    else:
        interpret = "😐 Sentimento indeterminado ou neutro."
    return label, score, interpret

# Streamlit App
st.set_page_config(page_title="IA Sentimento", layout="centered")
st.title("🌟 IA Sentimento (PT) 🌟")

analisador = obter_analisador()

st.sidebar.header("Opções")
modo = st.sidebar.radio("Escolha o modo de análise:", ("Analisar Texto", "Analisar Arquivo"))

if modo == "Analisar Texto":
    texto = st.text_area("Digite ou cole seu texto:", height=150)
    if st.button("Analisar"):  
        if texto.strip():
            label, score, interpret = analisar_texto(analisador, texto)
            timestamp = datetime.now().isoformat()
            salvar_resultado(timestamp, texto, label, score)
            st.markdown(f"**Sentimento detectado:** {label} (confiança: {score:.2f})")
            st.markdown(f"**Interpretação:** {interpret}")
        else:
            st.warning("Por favor, insira algum texto antes de analisar.")

elif modo == "Analisar Arquivo":
    uploaded_file = st.file_uploader("Envie um arquivo .txt com uma frase por linha", type=["txt"] )
    if uploaded_file:
        linhas = [l.decode('utf-8').strip() for l in uploaded_file.readlines() if l.strip()]
        if st.button("Processar Arquivo"):
            st.info(f"Processando {len(linhas)} frases...")
            for texto in linhas:
                label, score, interpret = analisar_texto(analisador, texto)
                timestamp = datetime.now().isoformat()
                salvar_resultado(timestamp, texto, label, score)
            st.success("Análise concluída! Resultados salvos em resultados.csv")
            st.download_button(
                label="Baixar resultados",
                data=open(CSV_FILE, 'rb').read(),
                file_name=CSV_FILE,
                mime='text/csv'
            )



