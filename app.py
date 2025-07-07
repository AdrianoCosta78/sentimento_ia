import streamlit as st
from pysentimiento import create_analyzer
import pandas as pd
from datetime import datetime

# Cria o analisador de sentimento em PT
analisador = create_analyzer(task="sentiment", lang="pt")

# Thresholds
THRESHOLD_POS = 0.85
THRESHOLD_NEG = 0.85

st.set_page_config(page_title="Sentimento IA", layout="centered")

st.title("📝 Sentimento IA (Streamlit)")
st.write("Cole um texto ou faça upload de um arquivo `.txt` com uma frase por linha, e veja o sentimento.")

# Área de texto livre
texto = st.text_area("Digite ou cole seu texto aqui:")

# Upload de arquivo de texto
uploaded_file = st.file_uploader("Ou faça upload de um arquivo .txt", type="txt")

# Botão para análise
if st.button("Analisar"):
    resultados = []

    def analisar_frase(frase):
        res = analisador.predict(frase)
        label = res.output      # 'POS', 'NEG' ou 'NEU'
        score = res.probas[label]
        if label == "POS" and score >= THRESHOLD_POS:
            categ = "Amigável"
        elif label == "NEG" and score >= THRESHOLD_NEG:
            categ = "Bravo"
        elif label == "NEU":
            categ = "Neutro"
        else:
            categ = "Indeterminado"
        return {
            "timestamp": datetime.now().isoformat(),
            "texto": frase,
            "label_raw": label,
            "score": round(score, 4),
            "categoria": categ
        }

    # Se recebeu texto
    if texto:
        resultados.append(analisar_frase(texto))

    # Se recebeu arquivo
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8").splitlines()
        for linha in content:
            if linha.strip():
                resultados.append(analisar_frase(linha.strip()))

    if not resultados:
        st.warning("Nenhum texto para analisar! Cole algo ou faça upload de um arquivo.")
    else:
        # Cria DataFrame aqui dentro
        df = pd.DataFrame(resultados)

        # Exibe resultados destacados
        for res in resultados:
            texto_mostrar = res["texto"]
            categoria = res["categoria"]
            score = res["score"]

            if categoria == "Amigável":
                st.success(f"😀 **Amigável** (confiança: {score:.2f})\n\n> {texto_mostrar}")
            elif categoria == "Bravo":
                st.error(f"😠 **Bravo** (confiança: {score:.2f})\n\n> {texto_mostrar}")
            elif categoria == "Neutro":
                st.info(f"😐 **Neutro** (confiança: {score:.2f})\n\n> {texto_mostrar}")
            else:
                st.warning(f"🤔 **Indeterminado** (confiança: {score:.2f})\n\n> {texto_mostrar}")
    
        # Botão para baixar CSV
        # csv = df.to_csv(index=False).encode('utf-8')
        # st.download_button(
        #     label="⬇️ Baixar resultados (CSV)",
        #     data=csv,
        #     file_name="resultados_sentimento.csv",
        #     mime="text/csv"
        #)

