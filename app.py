import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Universo da IA - 10 Exemplos Práticos",
    page_icon="🤖",
    layout="centered",
)

# Título Principal do Aplicativo
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

# =====================================================================
# 1. IA DAS NOTAS ESCOLARES
# =====================================================================
if opcao == "1. IA das Notas Escolares":
    st.header("📚 Contexto 1: IA das Notas Escolares")
    st.write("**Objetivo:** Prever a nota do aluno baseado nas horas de estudo.")

    estudos = pd.DataFrame(
        {"notas": [1, 2, 4, 6, 8, 10], "horas": [2, 4, 5, 7, 9, 10]}
    )
    st.write("Dados de exemplo usados para o treino:", estudos)

    X = estudos[["horas"]]
    y = estudos["notas"]
    modelo = LinearRegression().fit(X, y)

    horas_teste = st.slider(
        "Quantas horas você quer simular de estudo?",
        min_value=0,
        max_value=15,
        value=6,
    )
    # Criando um DataFrame idêntico ao de treino para evitar o erro de memória
    X_teste = pd.DataFrame({"horas": [horas_teste]})
    previsao = modelo.predict(X_teste)

    st.success(
        f"📝 **Previsão da IA:** Se o aluno estudar **{horas_teste} horas**, a nota estimada será **{previsao[0]:.2f}**."
    )

# =====================================================================
# 2. DETECTOR DE SONO GAMER
# =====================================================================
elif opcao == "2. Detector de Sono Gamer":
    st.header("🎮 Contexto 2: Detector de Sono Gamer")
    st.write(
        "**Objetivo:** Prever o nível de cansaço baseado nas horas jogando."
    )

    gamer = pd.DataFrame(
        {"horas_jogo": [1, 2, 4, 6, 8, 10], "cansaco": [1, 2, 3, 5, 8, 10]}
    )
    st.write("Dados de exemplo usados para o treino:", gamer)

    X = gamer[["horas_jogo"]]
    y = gamer["cansaco"]
    modelo = LinearRegression().fit(X, y)

    horas_teste = st.slider(
        "Quantas horas jogando?", min_value=0, max_value=15, value=5
    )
    X_teste = pd.DataFrame({"horas_jogo": [horas_teste]})
    previsao = modelo.predict(X_teste)

    st.warning(
        f"🥱 **Previsão da IA:** Jogando por **{horas_teste} horas**, o nível de cansaço estimado é **{previsao[0]:.2f}/10**."
    )

# =====================================================================
# 3. IA DO SORVETE
# =====================================================================
elif opcao == "3. IA do Sorvete":
    st.header("🍦 Contexto 3: IA do Sorvete")
    st.write(
        "**Objetivo:** Prever a quantidade de sorvetes vendidos pela temperatura ambiente."
    )

    sorvete = pd.DataFrame(
        {
            "temperatura": [18, 20, 24, 27, 30, 35],
            "vendas": [20, 25, 40, 55, 70, 100],
        }
    )
    st.write("Dados de exemplo usados para o treino:", sorvete)

    X = sorvete[["temperatura"]]
    y = sorvete["vendas"]
    modelo = LinearRegression().fit(X, y)

    temp_teste = st.slider(
        "Qual é a temperatura do dia (°C)?", min_value=10, max_value=45, value=25
    )
    X_teste = pd.DataFrame({"temperatura": [temp_teste]})
    previsao = modelo.predict(X_teste)

    st.info(
        f"☀️ **Previsão da IA:** Com uma temperatura de **{temp_teste}°C**, a previsão é vender aproximadamente **{previsao[0]:.0f} sorvetes**."
    )

# =====================================================================
# 4. DETECTOR DE APROVAÇÃO NINJA
# =====================================================================
elif opcao == "4. Detector de Aprovação Ninja":
    st.header("🥷 Contexto 4: Detector de Aprovação Ninja")
    st.write(
        "**Objetivo:** Classificar se o aluno será aprovado (1) ou reprovado (0) com base em suas faltas."
    )

    alunos = pd.DataFrame(
        {"faltas": [0, 1, 2, 5, 7, 10], "resultado": [1, 1, 1, 0, 0, 0]}
    )
    st.write("Dados de exemplo usados para o treino:", alunos)

    X = alunos[["faltas"]]
    y = alunos["resultado"]
    modelo = LogisticRegression().fit(X, y)

    faltas_teste = st.slider(
        "Número de faltas do aluno:", min_value=0, max_value=15, value=3
    )
    X_teste = pd.DataFrame({"faltas": [faltas_teste]})
    previsao = modelo.predict(X_teste)
    status = "Aprovado ✅" if previsao[0] == 1 else "Reprovado ❌"

    if previsao[0] == 1:
        st.success(
            f"Resultados com **{faltas_teste} faltas**: A IA classificou o aluno como **{status}**."
        )
    else:
        st.error(
            f"Resultados com **{faltas_teste} faltas**: A IA classificou o aluno como **{status}**."
        )

# =====================================================================
# 5. IA DO PET FELIZ
# =====================================================================
elif opcao == "5. IA do Pet Feliz":
    st.header("🐶 Contexto 5: IA do Pet Feliz")
    st.write("**Objetivo:** Prever a felicidade do cachorro usando passeios.")

    pets = pd.DataFrame(
        {"passeios": [1, 2, 3, 4, 5], "felicidade": [2, 4, 5, 8, 10]}
    )
    st.write("Dados de exemplo usados para o treino:", pets)

    X = pets[["passeios"]]
    y = pets["felicidade"]
    modelo = LinearRegression().fit(X, y)

    passeios_teste = st.slider(
        "Quantos passeios o pet vai dar hoje?", min_value=0, max_value=7, value=3
    )
    X_teste = pd.DataFrame({"passeios": [passeios_teste]})
    previsao = modelo.predict(X_teste)

    st.success(
        f"🐾 **Previsão da IA:** Dando **{passeios_teste} passeios**, o nível de felicidade do seu pet será de **{previsao[0]:.1f}/10**."
    )

# =====================================================================
# 6. DETECTOR DE FILME BOM
# =====================================================================
elif opcao == "6. Detector de Filme Bom":
    st.header("🎬 Contexto 6: Detector de Filme Bom")
    st.write("**Objetivo:** Prever a nota do filme usando a duração dele.")

    filmes = pd.DataFrame(
        {"duracao": [80, 90, 100, 110, 120], "nota": [4, 5, 7, 8, 9]}
    )
    st.write("Dados de exemplo usados para o treino:", filmes)

    X = filmes[["duracao"]]
    y = filmes["nota"]
    modelo = LinearRegression().fit(X, y)

    duracao_teste = st.slider(
        "Qual a duração do filme (em minutos)?",
        min_value=60,
        max_value=180,
        value=105,
    )
    X_teste = pd.DataFrame({"duracao": [duracao_teste]})
    previsao = modelo.predict(X_teste)

    st.info(
        f"🍿 **Previsão da IA:** Um filme de **{duracao_teste} minutos** deve receber a nota estimada de **{previsao[0]:.1f}/10**."
    )

# =====================================================================
# 7. IA DA PIZZA
# =====================================================================
elif opcao == "7. IA da Pizza":
    st.header("🍕 Contexto 7: IA da Pizza")
    st.write("**Objetivo:** Prever o preço da pizza pelo seu tamanho em cm.")

    pizza = pd.DataFrame(
        {"tamanho": [20, 25, 30, 35, 40], "preco": [20, 30, 40, 50, 60]}
    )
    st.write("Dados de exemplo usados para o treino:", pizza)

    X = pizza[["tamanho"]]
    y = pizza["preco"]
    modelo = LinearRegression().fit(X, y)

    tamanho_teste = st.slider(
        "Qual o tamanho da pizza (diâmetro em cm)?",
        min_value=15,
        max_value=50,
        value=32,
    )
    X_teste = pd.DataFrame({"tamanho": [tamanho_teste]})
    previsao = modelo.predict(X_teste)

    st.success(
        f"💵 **Previsão da IA:** Uma pizza com tamanho de **{tamanho_teste} cm** deve custar por volta de **R$ {previsao[0]:.2f}**."
    )

# =====================================================================
# 8. DETECTOR DE MÚSICA VIRAL
# =====================================================================
elif opcao == "8. Detector de Música Viral":
    st.header("🎵 Contexto 8: Detector de Música Viral")
    st.write(
        "**Objetivo:** Prever o potencial ou chance de uma música viralizar dado o seu ritmo (BPM)."
    )

    musica = pd.DataFrame(
        {"bpm": [80, 90, 100, 120, 140], "viral": [1, 2, 4, 7, 10]}
    )
    st.write("Dados de exemplo usados para o treino:", musica)

    X = musica[["bpm"]]
    y = musica["viral"]
    modelo = LinearRegression().fit(X, y)

    bpm_teste = st.slider(
        "Quantos BPM (Batidas Por Minuto) tem a música?",
        min_value=60,
        max_value=200,
        value=110,
    )
    X_teste = pd.DataFrame({"bpm": [bpm_teste]})
    previsao = modelo.predict(X_teste)

    st.warning(
        f"🚀 **Previsão da IA:** Uma música com **{bpm_teste} BPM** terá um índice de viralização de **{previsao[0]:.1f}/10**."
    )

# =====================================================================
# 9. IA DA ENERGIA DO CAFÉ
# =====================================================================
elif opcao == "9. IA da Energia do Café":
    st.header("☕ Contexto 9: IA da Energia do Café")
    st.write("**Objetivo:** Prever o nível de energia baseado em xícaras tomadas.")

    cafe = pd.DataFrame({"xicaras": [1, 2, 3, 4, 5], "energia": [2, 4, 6, 8, 10]})
    st.write("Dados de exemplo usados para o treino:", cafe)

    X = cafe[["xicaras"]]
    y = cafe["energia"]
    modelo = LinearRegression().fit(X, y)

    xicaras_teste = st.slider(
        "Quantas xícaras de café tomadas?", min_value=0.0, max_value=10.0, value=3.5
    )
    X_teste = pd.DataFrame({"xicaras": [xicaras_teste]})
    previsao = modelo.predict(X_teste)

    st.info(
        f"⚡ **Previsão da IA:** Tomando **{xicaras_teste} xícaras** de café, o nível de energia previsto é de **{previsao[0]:.1f}**."
    )

# =====================================================================
# 10. REDE NEURAL DOS SUPER-HERÓIS
# =====================================================================
elif opcao == "10. Rede Neural dos Super-Heróis":
    st.header("🦸‍♂️ Contexto 10: Rede Neural dos Super-Heróis")
    st.write(
        "**Objetivo:** Classificar se um herói é Forte (1) ou Fraco (0) com base no seu nível bruto de força."
    )

    herois = pd.DataFrame(
        {"forca": [1, 2, 3, 7, 8, 10], "heroi": [0, 0, 0, 1, 1, 1]}
    )
    st.write("Dados de exemplo usados para o treino:", herois)

    X = herois[["forca"]]
    y = herois["heroi"]
    modelo = LogisticRegression().fit(X, y)

    forca_teste = st.slider(
        "Escolha o nível de força do Herói:", min_value=1, max_value=15, value=5
    )
    X_teste = pd.DataFrame({"forca": [forca_teste]})
    previsao = modelo.predict(X_teste)
    status = "Forte 💪" if previsao[0] == 1 else "Fraco 🔍"

    if previsao[0] == 1:
        st.success(
            f"💥 **Previsão da IA:** Um herói com força **{forca_teste}** foi classificado como **{status}**."
        )
    else:
        st.error(
            f"💥 **Previsão da IA:** Um herói com força **{forca_teste}** foi classificado como **{status}**."
        )
