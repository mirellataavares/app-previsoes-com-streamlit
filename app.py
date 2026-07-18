import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import streamlit as st


st.set_page_config(
    page_title="Universo da IA - 10 Exemplos Práticos",
    page_icon="🤖",
    layout="centered",
)


st.title("🤖 O Universo da Inteligência Artificial")
st.markdown(
    "Bem-vindo! Este aplicativo foi feito para pessoas curiosas e comprometidas em entender como as máquinas aprendem. "
    "Use o menu ao lado para navegar entre **10 exemplos práticos** diferentes!"
)

st.sidebar.header("Escolha o Contexto")
opcao = st.sidebar.radio(
    "Selecione uma IA para testar:",
    [
        "1. IA das Notas Escolares",
        "2. Detector de Sono Gamer",
        "3. IA do Sorvete",
        "4. Detector de Aprovação Ninja",
        "5. IA do Pet Feliz",
        "6. Detector de Filme Bom",
        "7. IA da Pizza",
        "8. Detector de Música Viral",
        "9. IA da Energia do Café",
        "10. Rede Neural dos Super-Heróis",
    ],
)

st.divider()



@st.cache_resource
def treinar_modelos_regressao(tipo_contexto):
    """Treina os modelos de Regressão Linear apenas uma vez e guarda na memória."""
    if tipo_contexto == 1:  # Notas Escolares
        df = pd.DataFrame(
            {"notas": [1, 2, 4, 6, 8, 10], "horas": [2, 4, 5, 7, 9, 10]}
        )
        return LinearRegression().fit(df[["horas"]], df["notas"]), df
    elif tipo_contexto == 2:  # Gamer
        df = pd.DataFrame(
            {"horas_jogo": [1, 2, 4, 6, 8, 10], "cansaco": [1, 2, 3, 5, 8, 10]}
        )
        return LinearRegression().fit(df[["horas_jogo"]], df["cansaco"]), df
    elif tipo_contexto == 3:  # Sorvete
        df = pd.DataFrame(
            {
                "temperatura": [18, 20, 24, 27, 30, 35],
                "vendas": [20, 25, 40, 55, 70, 100],
            }
        )
        return LinearRegression().fit(df[["temperatura"]], df["vendas"]), df
    elif tipo_contexto == 5:  # Pet Feliz
        df = pd.DataFrame(
            {"passeios": [1, 2, 3, 4, 5], "felicidade": [2, 4, 5, 8, 10]}
        )
        return LinearRegression().fit(df[["passeios"]], df["felicidade"]), df
    elif tipo_contexto == 6:  # Filme Bom
        df = pd.DataFrame(
            {"duracao": [80, 90, 100, 110, 120], "nota": [4, 5, 7, 8, 9]}
        )
        return LinearRegression().fit(df[["duracao"]], df["nota"]), df
    elif tipo_contexto == 7:  # Pizza
        df = pd.DataFrame(
            {"tamanho": [20, 25, 30, 35, 40], "preco": [20, 30, 40, 50, 60]}
        )
        return LinearRegression().fit(df[["tamanho"]], df["preco"]), df
    elif tipo_contexto == 8:  # Música Viral
        df = pd.DataFrame(
            {"bpm": [80, 90, 100, 120, 140], "viral": [1, 2, 4, 7, 10]}
        )
        return LinearRegression().fit(df[["bpm"]], df["viral"]), df
    elif tipo_contexto == 9:  # Café
        df = pd.DataFrame(
            {"xicaras": [1, 2, 3, 4, 5], "energia": [2, 4, 6, 8, 10]}
        )
        return LinearRegression().fit(df[["xicaras"]], df["energia"]), df


@st.cache_resource
def treinar_modelos_classificacao(tipo_contexto):
    """Treina os modelos de Classificação Logística apenas uma vez e guarda na memória."""
    if tipo_contexto == 4:  # Ninja
        df = pd.DataFrame(
            {"faltas": [0, 1, 2, 5, 7, 10], "resultado": [1, 1, 1, 0, 0, 0]}
        )
        return LogisticRegression().fit(df[["faltas"]], df["resultado"]), df
    elif tipo_contexto == 10:  # Heróis
        df = pd.DataFrame(
            {"forca": [1, 2, 3, 7, 8, 10], "heroi": [0, 0, 0, 1, 1, 1]}
        )
        return LogisticRegression().fit(df[["forca"]], df["heroi"]), df



if opcao == "1. IA das Notas Escolares":
    st.header("📚 Contexto 1: IA das Notas Escolares")
    st.write("**Objetivo:** Prever a nota do aluno baseado nas horas de estudo.")

    modelo, dados = treinar_modelos_regressao(1)
    st.write("Dados de exemplo usados para o treino:", dados)

    horas_teste = st.slider(
        "Quantas horas você quer simular de estudo?", 0, 15, 6
    )
    previsao = modelo.predict(pd.DataFrame({"horas": [horas_teste]}))
    st.success(
        f"📝 **Previsão da IA:** Se o aluno estudar **{horas_teste} horas**, a nota estimada será **{previsao[0]:.2f}**."
    )

elif opcao == "2. Detector de Sono Gamer":
    st.header("🎮 Contexto 2: Detector de Sono Gamer")
    st.write(
        "**Objetivo:** Prever o nível de cansaço baseado nas horas jogando."
    )

    modelo, dados = treinar_modelos_regressao(2)
    st.write("Dados de exemplo usados para o treino:", dados)

    horas_teste = st.slider("Quantas horas jogando?", 0, 15, 5)
    previsao = modelo.predict(pd.DataFrame({"horas_jogo": [horas_teste]}))
    st.warning(
        f"🥱 **Previsão da IA:** Jogando por **{horas_teste} horas**, o nível de cansaço estimado é **{previsao[0]:.2f}/10**."
    )

elif opcao == "3. IA do Sorvete":
    st.header("🍦 Contexto 3: IA do Sorvete")
    st.write(
        "**Objetivo:** Prever a quantidade de sorvetes vendidos pela temperatura ambiente."
    )

    modelo, dados = treinar_modelos_regressao(3)
    st.write("Dados de exemplo usados para o treino:", dados)

    temp_teste = st.slider("Qual é a temperatura do dia (°C)?", 10, 45, 25)
    previsao = modelo.predict(pd.DataFrame({"temperatura": [temp_teste]}))
    st.info(
        f"☀️ **Previsão da IA:** Com uma temperatura de **{temp_teste}°C**, a previsão é vender aproximadamente **{previsao[0]:.0f} sorvetes**."
    )

elif opcao == "4. Detector de Aprovação Ninja":
    st.header("🥷 Contexto 4: Detector de Aprovação Ninja")
    st.write(
        "**Objetivo:** Classificar se o aluno será aprovado (1) ou reprovado (0) com base em suas faltas."
    )

    modelo, dados = treinar_modelos_classificacao(4)
    st.write("Dados de exemplo usados para o treino:", dados)

    faltas_teste = st.slider("Número de faltas do aluno:", 0, 15, 3)
    previsao = modelo.predict(pd.DataFrame({"faltas": [faltas_teste]}))
    status = "Aprovado ✅" if previsao[0] == 1 else "Reprovado ❌"

    if previsao[0] == 1:
        st.success(
            f"Resultados com **{faltas_teste} faltas**: A IA classificou o aluno como **{status}**."
        )
    else:
        st.error(
            f"Resultados com **{faltas_teste} faltas**: A IA classificou o aluno como **{status}**."
        )

elif opcao == "5. IA do Pet Feliz":
    st.header("🐶 Contexto 5: IA do Pet Feliz")
    st.write("**Objetivo:** Prever a felicidade do cachorro usando passeios.")

    modelo, dados = treinar_modelos_regressao(5)
    st.write("Dados de exemplo usados para o treino:", dados)

    passeios_teste = st.slider("Quantos passeios o pet vai dar hoje?", 0, 7, 3)
    previsao = modelo.predict(pd.DataFrame({"passeios": [passeios_teste]}))
    st.success(
        f"🐾 **Previsão da IA:** Dando **{passeios_teste} passeios**, o nível de felicidade do seu pet será de **{previsao[0]:.1f}/10**."
    )

elif opcao == "6. Detector de Filme Bom":
    st.header("🎬 Contexto 6: Detector de Filme Bom")
    st.write("**Objetivo:** Prever a nota do filme usando a duração dele.")

    modelo, dados = treinar_modelos_regressao(6)
    st.write("Dados de exemplo usados para o treino:", dados)

    duracao_teste = st.slider("Qual a duração do filme (em minutos)?", 60, 180, 105)
    previsao = modelo.predict(pd.DataFrame({"duracao": [duracao_teste]}))
    st.info(
        f"🍿 **Previsão da IA:** Um filme de **{duracao_teste} minutos** deve receber a nota estimada de **{previsao[0]:.1f}/10**."
    )

elif opcao == "7. IA da Pizza":
    st.header("🍕 Contexto 7: IA da Pizza")
    st.write("**Objetivo:** Prever o preço da pizza pelo seu tamanho em cm.")

    modelo, dados = treinar_modelos_regressao(7)
    st.write("Dados de exemplo usados para o treino:", dados)

    tamanho_teste = st.slider("Qual o tamanho da pizza (diâmetro em cm)?", 15, 50, 32)
    previsao = modelo.predict(pd.DataFrame({"tamanho": [tamanho_teste]}))
    st.success(
        f"💵 **Previsão da IA:** Uma pizza com tamanho de **{tamanho_teste} cm** deve custar por volta de **R$ {previsao[0]:.2f}**."
    )

elif opcao == "8. Detector de Música Viral":
    st.header("🎵 Contexto 8: Detector de Música Viral")
    st.write(
        "**Objetivo:** Prever o potencial ou chance de uma música viralizar dado o seu ritmo (BPM)."
    )

    modelo, dados = treinar_modelos_regressao(8)
    st.write("Dados de exemplo usados para o treino:", dados)

    bpm_teste = st.slider("Quantos BPM (Batidas Por Minuto) tem a música?", 60, 200, 110)
    previsao = modelo.predict(pd.DataFrame({"bpm": [bpm_teste]}))
    st.warning(
        f"🚀 **Previsão da IA:** Uma música com **{bpm_teste} BPM** terá um índice de viralização de **{previsao[0]:.1f}/10**."
    )

elif opcao == "9. IA da Energia do Café":
    st.header("☕ Contexto 9: IA da Energia do Café")
    st.write("**Objetivo:** Prever o nível de energia baseado em xícaras tomadas.")

    modelo, dados = treinar_modelos_regressao(9)
    st.write("Dados de exemplo usados para o treino:", dados)

    xicaras_teste = st.slider("Quantas xícaras de café tomadas?", 0.0, 10.0, 3.5)
    previsao = modelo.predict(pd.DataFrame({"xicaras": [xicaras_teste]}))
    st.info(
        f"⚡ **Previsão da IA:** Tomando **{xicaras_teste} xícaras** de café, o nível de energia previsto é de **{previsao[0]:.1f}**."
    )

elif opcao == "10. Rede Neural dos Super-Heróis":
    st.header("🦸‍♂️ Contexto 10: Rede Neural dos Super-Heróis")
    st.write(
        "**Objetivo:** Classificar se um herói é Forte (1) ou Fraco (0) com base no seu nível bruto de força."
    )

    modelo, dados = treinar_modelos_classificacao(10)
    st.write("Dados de exemplo usados para o treino:", dados)

    forca_teste = st.slider("Escolha o nível de força do Herói:", 1, 15, 5)
    previsao = modelo.predict(pd.DataFrame({"forca": [forca_teste]}))
    status = "Forte 💪" if previsao[0] == 1 else "Fraco 🔍"

    if previsao[0] == 1:
        st.success(
            f"💥 **Previsão da IA:** Um herói com força **{forca_teste}** foi classificado como **{status}**."
        )
    else:
        st.error(
            f"💥 **Previsão da IA:** Um herói com força **{forca_teste}** foi classificado como **{status}**."
        )
