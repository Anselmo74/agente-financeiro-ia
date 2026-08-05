import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
import logging
import sqlite3
import plotly.graph_objects as go
from datetime import datetime

# Configuração de Layout da Página Web (Obrigatório ser o primeiro comando)
st.set_page_config(
    page_title="Agente IA Financeiro B3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Silenciar avisos e logs secundários
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =====================================================================
# CONFIGURAÇÕES GLOBAIS
# =====================================================================
URL_IA_PROXIMIDADE = "https://openrouter.ai"
API_KEY_IA = "fd10bd41-3d8f-50da-8a73-716eef2ec764"

RISCO_MAXIMO_FINANCEIRO = 1000.00
LIMITE_LIQUIDEZ_DIARIA = 1000000.00

DB_NAME = "trades_historico.db"

def inicializar_banco():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historico_sinais (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            ticker TEXT,
            estrategia TEXT,
            preco_entrada REAL,
            stop_loss REAL,
            alvo REAL,
            resultado TEXT DEFAULT 'Aberto'
        )
    """)
    conn.commit()
    conn.close()

def salvar_sinal_no_banco(ticker, estrategia, preco, stop, alvo):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    hoje = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT id FROM historico_sinais 
        WHERE ticker = ? AND estrategia = ? AND data_hora LIKE ?
    """, (ticker, estrategia, f"{hoje}%"))
    if cursor.fetchone() is None:
        data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO historico_sinais (data_hora, ticker, estrategia, preco_entrada, stop_loss, alvo)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (data_atual, ticker, estrategia, preco, stop, alvo))
        conn.commit()
    conn.close()

def carregar_historico_banco():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM historico_sinais ORDER BY id DESC", conn)
    conn.close()
    return df

inicializar_banco()

def obter_universo_b3():
    tickers_base = [
        "CMIN3", "UGPA3", "EMBR3", "VALE3", "PETR4", "ITUB4", "BBDC4", "BBAS3", 
        "WEGE3", "RENT3", "PRIO3", "SBSP3", "SUZB3", "JBSS3", "LREN3", "RAIL3",
        "ABEV3", "B3SA3", "BBSE3", "RADL3", "HAPV3", "GGBR4", "CSNA3", "MGLU3"
    ]
    return sorted(list(set([f"{t}.SA" for t in tickers_base])))

def calcular_ifr_professional(series, periodos=14):
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
    """Consulta o OpenRouter para interpretar o cenário do papel em 15 words."""
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
            return response.json()['choices']['message']['content'].strip()
    except: pass
    return f"Ajuste técnico de carteiras institucionais perto de R$ {preco:.2f}."

@st.cache_data(ttl=120)  # Reduzido para 2 minutos para maior dinamismo intraday
def processar_mercado_duplo():
    """Varre o universo selecionado da B3, classifica os ativos e salva no SQLite."""
    lista_ativos = obter_universo_b3()
    pool_exaustao = []
    pool_retomada = []

    for ticker in lista_ativos:
        try:
            # Varredura padrão em 15m para alimentar os rankings
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
                dist_stop = atr_atual * 2 if atr_atual > 0 else preco_atual * 0.02
                stop_loss = preco_atual - dist_stop
                alvo_lucro = preco_atual + (dist_stop * 1.5)
                
                salvar_sinal_no_banco(ticker.replace('.SA',''), "Exaustão de Venda", preco_atual, stop_loss, alvo_lucro)
                
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
                
                dist_stop = atr_atual * 1.5 if atr_atual > 0 else preco_atual * 0.015
                stop_loss = preco_atual - dist_stop
                alvo_lucro = preco_atual + (dist_stop * 2.0)
                
                salvar_sinal_no_banco(ticker.replace('.SA',''), "Retomada de Subida", preco_atual, stop_loss, alvo_lucro)
                
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
# INTERFACE VISUAL AVANÇADA (STREAMLIT APP UI)
# =====================================================================
st.title("🤖 AGENTE FINANCEIRO IA: Painel Quantitativo Avançado")
st.markdown("---")

# Abas principais da ferramenta
tab_monitoramento, tab_historico = st.tabs(["📊 Gráficos & Sinais Online", "🗄️ Histórico SQLite"])

with tab_monitoramento:
    col_esquerda, col_direita = st.columns([1, 1.8])

    with col_esquerda:
        st.subheader("🔍 Ativos Selecionados")
        st.caption("Clique na linha de qualquer tabela para carregar o gráfico instantaneamente.")

        with st.spinner("Rodando scanner de mercado..."):
            df_exaustao, df_retomada = processar_mercado_duplo()

        # Ativo padrão de fallback caso nada seja selecionado
        ativo_final = "EMBR3"

        # 1. Tabela Interativa de Retomada com Captura de Clique
        st.markdown("**🚀 Top 5 - Retomada Confirmada de Alta**")
        if not df_retomada.empty:
            sel_ret = st.dataframe(
                df_retomada[['Ativo', 'Preço (R$)', 'IFR', 'Vol_Ratio']], 
                use_container_width=True, hide_index=True,
                selection_mode="single-row", on_select="rerun"
            )
            # Se o usuário clicar em uma linha, captura o ativo correspondente
            if sel_ret.get("selection") and sel_ret["selection"]["rows"]:
                idx_linha = sel_ret["selection"]["rows"]
                ativo_final = df_retomada.iloc[idx_linha]['Ativo']
        else:
            st.info("Nenhuma ação em reversão de alta.")

        # 2. Tabela Interativa de Exaustão com Captura de Clique
        st.markdown("**💥 Top 5 - Clímax / Exaustão de Venda**")
        if not df_exaustao.empty:
            sel_ex = st.dataframe(
                df_exaustao[['Ativo', 'Preço (R$)', 'IFR', 'Vol_Ratio']], 
                use_container_width=True, hide_index=True,
                selection_mode="single-row", on_select="rerun"
            )
            if sel_ex.get("selection") and sel_ex["selection"]["rows"]:
                idx_linha = sel_ex["selection"]["rows"]
                ativo_final = df_exaustao.iloc[idx_linha]['Ativo']
        else:
            st.info("Nenhuma ação em pânico institucional.")

    with col_direita:
        st.subheader(f"📊 Análise Visual do Preço: {ativo_final}")

        # Controles Avançados do Gráfico solicitados pelo usuário
        c1, c2 = st.columns(2)
        with c1:
            periodo_opcao = st.selectbox("Período Histórico:", ["1 dia (Intraday)", "Últimas Horas", "5 dias", "1 mês"])
        with c2:
            candle_opcao = st.selectbox("Tempo do Candle (Tempo Gráfico):", ["15 minutos", "5 minutos", "30 minutos", "1 hora", "1 dia", "1 semana"], index=0)

        # Mapeamento de Parâmetros do Yahoo Finance com base nas opções selecionadas
        map_periodo = {"1 dia (Intraday)": "1d", "Últimas Horas": "1d", "5 dias": "5d", "1 mês": "1mo"}
        map_candle = {"5 minutos": "5m", "15 minutos": "15m", "30 minutos": "30m", "1 hora": "1h", "1 dia": "1d", "1 semana": "1wk"}

        periodo_yf = map_periodo[periodo_opcao]
        candle_yf = map_candle[candle_opcao]

        try:
            ticker_yf = f"{ativo_final}.SA"
            dados = yf.download(ticker_yf, period=periodo_yf, interval=candle_yf, progress=False, auto_adjust=True, multi_level_index=False)

            if not dados.empty:
                dados = dados.dropna(subset=['Close'])

                # Filtro específico para a opção "Últimas Horas" (pega apenas os últimos 16 candles do dia)
                if periodo_opcao == "Últimas Horas" and len(dados) > 16:
                    dados = dados.tail(16)

                preco_atual = float(dados['Close'].iloc[-1])

                # Parâmetros de risco dinâmicos via ATR para as linhas guias do gráfico
                high_low = dados['High'] - dados['Low']
                atr_calc = float(high_low.rolling(window=14).mean().fillna(preco_atual * 0.01).iloc[-1])
                
                stop_loss = preco_atual - (atr_calc * 2)
                alvo_lucro = preco_atual + (atr_calc * 1.5)
                quantidade_lote = int(RISCO_MAXIMO_FINANCEIRO / (preco_atual - stop_loss)) if (preco_atual - stop_loss) > 0 else 0

                # Adiciona Média Móvel de Referência de 20 períodos para apoio visual à decisão
                dados['Média Ref (20)'] = dados['Close'].rolling(window=20).mean().fillna(dados['Close'])

                # Métricas Rápidas na Tela
                m1, m2, m3 = st.columns(3)
                m1.metric("Preço Atual", f"R$ {preco_atual:.2f}")
                m2.metric("Stop Loss Recomendado", f"R$ {stop_loss:.2f}")
                m3.metric("Alvo do Trade", f"R$ {alvo_lucro:.2f}")

                # -----------------------------------------------------------------
                # CONSTRUÇÃO DO GRÁFICO PROFISSIONAL (PLOTLY)
                # -----------------------------------------------------------------
                fig = go.Figure()

                # Linha de Preço principal
                fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço Fechamento', line=dict(color='#2ca02c', width=2.5)))
                
                # Linha da Média Móvel de Apoio
                fig.add_trace(go.Scatter(x=dados.index, y=dados['Média Ref (20)'], name='Média Móvel (20)', line=dict(color='#ff7f0e', width=1.5, dash='solid')))

                # Linhas Horizontais Estáticas de Alvo e Stop Loss para apoiar a decisão visual
                fig.add_hline(y=alvo_lucro, line_dash="dash", line_color="#2ca02c", annotation_text="Alvo Lucro", annotation_position="top right")
                fig.add_hline(y=stop_loss, line_dash="dash", line_color="#d62728", annotation_text="Stop Loss", annotation_position="bottom right")

                # ESTRELA DO GRÁFICO: Remoção cirúrgica dos horários em que o mercado fica parado
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]), # Oculta os finais de semana (Sábado e Domingo)
                        dict(bounds=[18, 10], pattern="hour") # Oculta o período das 18:00 até às 10:00 da manhã seguinte
                    ]
                )

                # Customização Estética do Layout para Modo Dark Elegante
                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"🛡️ **Gestão de Posição:** Opere no máximo **{quantidade_lote} ações** para manter o risco fixado em R$ {RISCO_MAXIMO_FINANCEIRO:.2f}.")

                # CORREÇÃO EFETUADA: Mudança de 'St' para 'st' minúsculo
                with st.spinner("Interpretando fatos de mercado..."):
                    feed = buscar_noticias_reais_yfinance(ticker_yf)
                    contexto_ia = gerar_fato_ocorrido_por_ia(ativo_final, preco_atual, feed)
                st.info(f"📰 **Contexto IA:** {contexto_ia}")

            else:
                st.error("Sem dados de cotação para as combinações gráficas selecionadas.")
        except Exception as e:
            st.error(f"Erro ao renderizar painel visual: {str(e)}")

# ---------------------------------------------------------------------
# ABA 2: HISTÓRICO DE SINAIS (AUDITORIA SQLITE)
# ---------------------------------------------------------------------
with tab_historico:
    st.subheader("🗄️ Histórico de Varreduras do Robô")
    df_db = carregar_historico_banco()
    if not df_db.empty:
        df_db.columns = ['ID', 'Data/Hora', 'Ativo', 'Estratégia', 'Preço Entrada', 'Stop Loss', 'Alvo', 'Status']
        st.dataframe(df_db, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum registro gravado nas tabelas locais até o momento.")

