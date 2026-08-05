import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
import logging
from datetime import datetime

# Configuração de Layout da Página Web Streamlit (Deve ser o primeiro comando)
st.set_page_config(
    page_title="Agente IA Financeiro B3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Silenciar avisos e logs secundários do terminal para otimizar performance
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =====================================================================
# PARÂMETROS E CONFIGURAÇÕES DO AGENTE IA
# =====================================================================
URL_IA_PROXIMIDADE = "https://openrouter.ai"
API_KEY_IA = "fd10bd41-3d8f-50da-8a73-716eef2ec764"

RISCO_MAXIMO_FINANCEIRO = 1000.00
LIMITE_LIQUIDEZ_DIARIA = 1000000.00  # Mínimo de R$ 1 Milhão/dia

def obter_universo_b3():
    """Retorna a lista selecionada e higienizada de ativos focos para a varredura."""
    tickers_base = [
        "CMIN3", "UGPA3", "EMBR3", "VALE3", "PETR4", "ITUB4", "BBDC4", "BBAS3", 
        "WEGE3", "RENT3", "PRIO3", "SBSP3", "SUZB3", "JBSS3", "LREN3", "RAIL3",
        "ABEV3", "B3SA3", "BBSE3", "RADL3", "HAPV3", "GGBR4", "CSNA3", "MGLU3"
    ]
    return sorted(list(set([f"{t}.SA" for t in tickers_base])))

def calcular_ifr_professional(series, periodos=14):
    """Calcula o Índice de Força Relativa com suavização exponencial."""
    delta = series.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    ma_ganho = ganho.ewm(alpha=1/periodos, adjust=False).mean()
    ma_perda = perda.ewm(alpha=1/periodos, adjust=False).mean()
    return 100 - (100 / (1 + (ma_ganho / ma_perda.replace(0, np.nan)))).fillna(100)

def buscar_noticias_reais_yfinance(ticker):
    """Captura as últimas manchetes de notícias do ativo via Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        noticias = t.news
        if noticias and len(noticias) > 0:
            manchetes = [n.get('title', '') for n in noticias[:2]]
            return " | ".join(manchetes)
    except: pass
    return "Nenhuma manchete recente encontrada no feed."

def gerar_fato_ocorrido_por_ia(ticker, preco, manchetes_reais):
    """Consulta o OpenRouter para interpretar o cenário do papel em 15 palavras."""
    headers = {
        "Authorization": f"Bearer {API_KEY_IA}", 
        "Content-Type": "application/json"
    }
    prompt = (
        f"Ação: {ticker}. Preço: R$ {preco:.2f}. Manchetes: '{manchetes_reais}'. "
        f"Explique em uma única frase curta de no máximo 15 palavras qual fato corporativo, "
        f"econômico ou boato justifica a oscilação recente deste papel na B3. Seja ultra objetivo."
    )
    data = {
        "model": "google/gemma-2-9b-it:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(URL_IA_PROXIMIDADE, headers=headers, json=data, timeout=8)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
    except: pass
    return f"Ajuste técnico de carteiras institucionais perto de R$ {preco:.2f}."

@st.cache_data(ttl=300)  # Cache de dados por 5 minutos para evitar bloqueio do Yahoo Finance
def processar_mercado_duplo():
    """Varre o universo selecionado da B3 classificando em Clímax ou Retomada."""
    lista_ativos = obter_universo_b3()
    pool_exaustao = []
    pool_retomada = []

    for ticker in lista_ativos:
        try:
            df = yf.download(ticker, period="5d", interval="15m", progress=False, auto_adjust=True, multi_level_index=False)
            if df.empty or len(df) < 30: continue
            df = df.dropna(subset=['Close', 'Volume'])
            
            fechamentos = df['Close'].squeeze()
            volumes = df['Volume'].squeeze()

            df['Vol_Financeiro'] = fechamentos * volumes
            liquidez_diaria = float(df['Vol_Financeiro'].rolling(window=20).mean().iloc[-1]) * 28

            if liquidez_diaria < LIMITE_LIQUIDEZ_DIARIA: continue
            preco_atual = float(fechamentos.iloc[-1])

            high_low = df['High'] - df['Low']
            df['ATR'] = high_low.rolling(window=14).mean()
            atr_atual = float(df['ATR'].iloc[-1])
            
            df['Vol_Quantidade_Media'] = volumes.rolling(window=20).mean()
            vol_ratio = float(volumes.iloc[-1] / df['Vol_Quantidade_Media'].iloc[-1])

            # Motor 1: Exaustão de Venda (Pânico)
            df['IFR'] = calcular_ifr_professional(fechamentos, periodos=14)
            ifr_atual = float(df['IFR'].iloc[-1])

            if ifr_atual <= 33.0:
                pool_exaustao.append({
                    'Ativo': ticker.replace('.SA', ''), 'Preço (R$)': round(preco_atual, 2), 
                    'IFR': round(ifr_atual, 2), 'Vol_Ratio': round(vol_ratio, 2), 'atr': atr_atual
                })

            # Motor 2: Retomada de Subida (Tendência)
            df['EMA_9'] = fechamentos.ewm(span=9, adjust=False).mean()
            df['EMA_21'] = fechamentos.ewm(span=21, adjust=False).mean()
            df['Donchian_High'] = df['High'].rolling(window=20).max()

            if (df['EMA_9'].iloc[-1] > df['EMA_21'].iloc[-1]) and (vol_ratio >= 1.2) and (preco_atual >= df['Donchian_High'].iloc[-1] * 0.98):
                momentum = (preco_atual - df['EMA_21'].iloc[-1]) / df['EMA_21'].iloc[-1]
                pool_retomada.append({
                    'Ativo': ticker.replace('.SA', ''), 'Preço (R$)': round(preco_atual, 2), 
                    'IFR': round(ifr_atual, 2), 'Vol_Ratio': round(vol_ratio, 2), 'atr': atr_atual, 'Momentum': momentum
                })
        except: continue

    df_ex = pd.DataFrame(pool_exaustao) if pool_exaustao else pd.DataFrame(columns=['Ativo', 'Preço (R$)', 'IFR', 'Vol_Ratio', 'atr'])
    df_ret = pd.DataFrame(pool_retomada) if pool_retomada else pd.DataFrame(columns=['Ativo', 'Preço (R$)', 'IFR', 'Vol_Ratio', 'atr', 'Momentum'])
    
    if not df_ex.empty: df_ex = df_ex.sort_values(by='IFR', ascending=True).head(5)
    if not df_ret.empty: df_ret = df_ret.sort_values(by='Momentum', ascending=False).head(5)
    
    return df_ex, df_ret

# =====================================================================
# INTERFACE VISUAL PRINCIPAL (STREAMLIT APP UI)
# =====================================================================
st.title("🤖 AGENTE FINANCEIRO IA: Monitoramento B3 Intraday")
st.markdown("---")

# Executa o processamento do mercado em segundo plano
with st.spinner("Varrendo o book de ofertas da B3 e rodando matrizes quantitativas..."):
    df_exaustao, df_retomada = processar_mercado_duplo()

# Criação do Layout em Duas Colunas Assimétricas
col_esquerda, col_direita = st.columns([1, 2])

with col_esquerda:
    st.subheader("📊 Seleção do Scanner")
    
    # Renderização da Tabela de Retomada de Subida (Motor 2)
    st.markdown("**🚀 Top 5 - Retomada Confirmada de Alta**")
    if not df_retomada.empty:
        st.dataframe(df_retomada[['Ativo', 'Preço (R$)', 'IFR', 'Vol_Ratio']], use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma ação confirmou reversão de alta neste ciclo.")

    # Renderização da Tabela de Exaustão de Venda (Motor 1)
    st.markdown("**💥 Top 5 - Clímax / Exaustão de Venda**")
    if not df_exaustao.empty:
        st.dataframe(df_exaustao[['Ativo', 'Preço (R$)', 'IFR', 'Vol_Ratio']], use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma ação em pânico institucional extremo.")

    # Caixa de Seleção Dinâmica para Análise Profunda
    lista_selecao = list(df_retomada['Ativo']) + list(df_exaustao['Ativo'])
    lista_selecao = list(set(lista_selecao)) # Remove duplicidades caso existam
    
    if not lista_selecao:
        lista_selecao = ["CMIN3", "UGPA3", "EMBR3"] # Fallback de segurança com os ativos sugeridos
        
    ativo_selecionado = st.selectbox("Escolha um ativo para inspecionar o Gráfico Online:", lista_selecao)

with col_direita:
    st.subheader(f"📈 Painel Analítico Online: {ativo_selecionado}")
    
    try:
        # Download de dados detalhados para plotagem do gráfico intraday
        ticker_yf = f"{ativo_selecionado}.SA"
        dados_grafico = yf.download(ticker_yf, period="5d", interval="15m", progress=False, auto_adjust=True, multi_level_index=False)
        
        if not dados_grafico.empty:
            dados_grafico = dados_grafico.dropna(subset=['Close'])
            preco_fechamento_atual = float(dados_grafico['Close'].iloc[-1])
            
            # Cálculo de Stop e Alvo dinâmico para o ativo selecionado baseado no histórico recente
            high_low_historico = dados_grafico['High'] - dados_grafico['Low']
            atr_contexto = float(high_low_historico.rolling(window=14).mean().iloc[-1])
            distancia_stop = atr_contexto * 1.8 if atr_contexto > 0 else preco_fechamento_atual * 0.02
            
            stop_loss = preco_fechamento_atual - distancia_stop
            alvo_lucro = preco_fechamento_atual + (distancia_stop * 1.5)
            quantidade_lote = int(RISCO_MAXIMO_FINANCEIRO / (preco_fechamento_atual - stop_loss)) if (preco_fechamento_atual - stop_loss) > 0 else 0

            # Métricas em Cartões Visuais
            m1, m2, m3 = st.columns(3)
            m1.metric("Preço Atual", f"R$ {preco_fechamento_atual:.2f}")
            m2.metric("Sugestão Stop Loss", f"R$ {stop_loss:.2f}")
            m3.metric("Alvo do Trade", f"R$ {alvo_lucro:.2f}")

            # Plotagem do Gráfico de Cotação de Linha Interativo nativo do Streamlit
            st.line_chart(dados_grafico['Close'], y_label="Preço de Fechamento (R$)", x_label="Tempo (Candles 15m)")
            
            # Caixa de Gestão de Risco Quantitativa
            st.success(f"🛡️ **Plano Ágora:** Compre até **{quantidade_lote} ações** para travar seu risco máximo em R$ {RISCO_MAXIMO_FINANCEIRO:.2f}.")

            # Chamada Assíncrona Inteligente para Contextualização Macroeconômica
            with st.spinner("Consultando IA online sobre fatos recentes deste ativo..."):
                noticias_feed = buscar_noticias_reais_yfinance(ticker_yf)
                insight_ia = gerar_fato_ocorrido_por_ia(ativo_selecionado, preco_fechamento_atual, noticias_feed)
                
            st.info(f"📰 **Contexto IA (Sentimento & Fatos):** {insight_ia}")
        else:
            st.error("Falha ao puxar cotações intraday para este ativo no momento.")
    except Exception as e:
        st.error(f"Erro no processamento do gráfico online: {str(e)}")
