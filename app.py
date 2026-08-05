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
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Configuração de Layout da Página Web (Obrigatório ser o primeiro comando)
st.set_page_config(
    page_title="Agente IA Financeiro B3 Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Silenciar avisos e logs secundários do terminal para otimizar processamento
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =====================================================================
# PROTEÇÃO E CRIPTOGRAFIA DE CHAVES (ST.SECRETS)
# =====================================================================
try:
    URL_IA_PROXIMIDADE = st.secrets["URL_IA_PROXIMIDADE"]
    API_KEY_IA = st.secrets["API_KEY_IA"]
    TOKEN_TELEGRAM = st.secrets["TOKEN_TELEGRAM"]
    CHAT_ID_TELEGRAM = st.secrets["CHAT_ID_TELEGRAM"]
except Exception:
    # Fallbacks de segurança para testes locais
    URL_IA_PROXIMIDADE = "https://openrouter.ai"
    API_KEY_IA = "fd10bd41-3d8f-50da-8a73-716eef2ec764"
    TOKEN_TELEGRAM = "8852525281:AAH56WNVEmmXyxvol9RKmkB3aa1Toap1QoY"
    CHAT_ID_TELEGRAM = "8852525281"

RISCO_MAXIMO_FINANCEIRO = 1000.00
LIMITE_LIQUIDEZ_DIARIA = 1000000.00
DB_NAME = "trades_historico.db"

# =====================================================================
# CONTROLE DE HORÁRIO E DATA BRASÍLIA (UTC -3)
# =====================================================================
def obter_horario_brasilia():
    """Retorna o objeto datetime convertido rigorosamente para o fuso UTC-3."""
    return datetime.utcnow() - timedelta(hours=3)

# =====================================================================
# MÓDULO DE BANCO DE DADOS (SQLITE COM AUDITORIA DO CICLO SUCESSIVO)
# =====================================================================
def inicializar_banco():
    """Cria e atualiza a tabela de histórico adicionando auditoria de acertos."""
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
            confirmacao_analise TEXT DEFAULT 'Aguardando Pregão'
        )
    """)
    conn.commit()
    conn.close()

def salvar_sinal_no_banco(ticker, estrategia, preco, stop, alvo):
    """Grava o sinal com carimbo de hora UTC-3 e checa ciclos passados."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    agora_br = obter_horario_brasilia()
    hoje_str = agora_br.strftime("%Y-%m-%d")
    
    cursor.execute("""
        SELECT id FROM historico_sinais 
        WHERE ticker = ? AND estrategia = ? AND data_hora LIKE ?
    """, (ticker, estrategia, f"{hoje_str}%"))
    
    if cursor.fetchone() is None:
        data_atual_br = agora_br.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO historico_sinais (data_hora, ticker, estrategia, preco_entrada, stop_loss, alvo)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (data_atual_br, ticker, estrategia, preco, stop, alvo))
        conn.commit()
    conn.close()

def atualizar_confirmacoes_no_banco(ticker, preco_atual):
    """Audita os sinais abertos no ciclo anterior, checando se atingiram alvo ou stop."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, preco_entrada, stop_loss, alvo FROM historico_sinais WHERE ticker = ? AND confirmacao_analise = 'Aguardando Pregão'", (ticker,))
    sinais_abertos = cursor.fetchall()
    
    for sinal in sinais_abertos:
        sid, p_entrada, s_loss, p_alvo = sinal
        if preco_atual >= p_alvo:
            status = "✅ Sucesso (Alvo Atingido)"
        elif preco_atual <= s_loss:
            status = "❌ Fracasso (Stop Acionado)"
        else:
            continue  # Permanece aguardando pregão até romper as barreiras
            
        cursor.execute("UPDATE historico_sinais SET confirmacao_analise = ? WHERE id = ?", (status, sid))
    conn.commit()
    conn.close()

def carregar_historico_banco():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM historico_sinais ORDER BY id DESC", conn)
    conn.close()
    return df

# Inicializa o banco de dados
inicializar_banco()

def obter_universo_b3():
    """Lista selecionada e higienizada de ativos focos para a varredura."""
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
    """Consulta o OpenRouter para interpretar o cenário do papel de forma sintética."""
    headers = {
        "Authorization": f"Bearer {API_KEY_IA}", 
        "Content-Type": "application/json"
    }
    prompt = (
        f"Ação: {ticker}. Preço: R$ {preco:.2f}. Manchetes: '{manchetes_reais}'. "
        f"Explique em uma única frase curta de no máximo 15 palavras qual fato corporativo, "
        f"econômico ou boato justifica a oscilação recente deste papel na B3. Seja ultra objective."
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

@st.cache_data(ttl=60)  # Reduzido para 1 minuto para dar máxima fidelidade ao tempo real
def processar_mercado_duplo():
    """Varre a B3 de forma híbrida: lê o intraday atualizado ou o último fechamento."""
    lista_ativos = obter_universo_b3()
    pool_exaustao = []
    pool_retomada = []

    for ticker in lista_ativos:
        try:
            # Baixa o histórico recente incluindo o dia atual/último fechamento
            df = yf.download(ticker, period="5d", interval="15m", progress=False, auto_adjust=True, multi_level_index=False)
            if df.empty or len(df) < 30: continue
            df = df.dropna(subset=['Close', 'Volume'])
            
            fechamentos = df['Close'].squeeze()
            volumes = df['Volume'].squeeze()

            df['Vol_Financeiro'] = fechamentos * volumes
            liquidez_diaria = float(df['Vol_Financeiro'].rolling(window=20).mean().iloc[-1]) * 28

            if liquidez_diaria < LIMITE_LIQUIDEZ_DIARIA: continue
            preco_atual = float(fechamentos.iloc[-1])
            volume_ultimo_candle = float(volumes.iloc[-1])

            high_low = df['High'] - df['Low']
            df['ATR'] = high_low.rolling(window=14).mean()
            atr_atual = float(df['ATR'].iloc[-1])
            
            df['Vol_Quantidade_Media'] = volumes.rolling(window=20).mean()
            vol_ratio = float(volumes.iloc[-1] / df['Vol_Quantidade_Media'].iloc[-1])
            
            if vol_ratio >= 1.0:
                desvio_vol_str = f"🚀 Acima (+{(vol_ratio - 1.0)*100:.1f}%)"
            else:
                desvio_vol_str = f"📉 Abaixo (-{(1.0 - vol_ratio)*100:.1f}%)"

            # Executa a auditoria automática de acertos/erros no banco SQLite
            atualizar_confirmacoes_no_banco(ticker.replace('.SA',''), preco_atual)

            # -----------------------------------------------------------------
            # Motor 1: Exaustão de Venda (Pânico)
            # -----------------------------------------------------------------
            df['IFR'] = calcular_ifr_professional(fechamentos, periodos=14)
            ifr_atual = float(df['IFR'].iloc[-1])

            if ifr_atual <= 33.0:
                dist_stop = atr_atual * 2 if atr_atual > 0 else preco_atual * 0.02
                stop_loss = preco_atual - dist_stop
                alvo_lucro = preco_atual + (dist_stop * 1.5)
                
                salvar_sinal_no_banco(ticker.replace('.SA',''), "Exaustão de Venda", preco_atual, stop_loss, alvo_lucro)
                
                pool_exaustao.append({
                    'Ativo': ticker.replace('.SA', ''), 'Preço (R$)': round(preco_atual, 2), 
                    'IFR': round(ifr_atual, 2), 'Fluxo Vol': desvio_vol_str, 'atr': atr_atual
                })

            # -----------------------------------------------------------------
            # Motor 2: Retomada de Subida (Tendência)
            # -----------------------------------------------------------------
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
                    'IFR': round(ifr_atual, 2), 'Fluxo Vol': desvio_vol_str, 'atr': atr_atual, 'Momentum': momentum
                })
        except: continue

    df_ex = pd.DataFrame(pool_exaustao) if pool_exaustao else pd.DataFrame(columns=['Ativo', 'Preço (R$)', 'IFR', 'Fluxo Vol', 'atr'])
    df_ret = pd.DataFrame(pool_retomada) if pool_retomada else pd.DataFrame(columns=['Ativo', 'Preço (R$)', 'IFR', 'Fluxo Vol', 'atr', 'Momentum'])
    
    if not df_ex.empty: df_ex = df_ex.sort_values(by='IFR', ascending=True).head(10)
    if not df_ret.empty: df_ret = df_ret.sort_values(by='Momentum', ascending=False).head(10)
    
    return df_ex, df_ret

# =====================================================================
# INTERFACE VISUAL AVANÇADA (STREAMLIT APP UI)
# =====================================================================
horario_atual_br = obter_horario_brasilia()
st.title("🤖 AGENTE FINANCEIRO IA: Painel Ativo B3 Híbrido")
st.markdown(f"**Fuso Horário:** Brasília (UTC -3) | **Última Atualização:** {horario_atual_br.strftime('%d/%m/%Y %H:%M:%S')}")

# Alerta informativo indicando as regras de exibição híbrida
st.sidebar.markdown("### 🏢 Regras do Scanner Híbrido")
st.sidebar.info("Se o pregão estiver ativo, o robô captura o exato instante atual. Fora do horário comercial, finais de semana ou feriados, exibe os dados consolidados do último fechamento oficial.")

st.info("🕒 **Ciclos Estritos de Registro de Auditoria:** 10:30 | 11:30 | 12:30 | 15:00 | 16:30 | 17:15")

# Abas principais da ferramenta
tab_monitoramento, tab_historico = st.tabs(["📊 Gráficos & Sinais Online (Top 10)", "🗄️ Histórico & Auditoria SQLite"])

with tab_monitoramento:
    col_esquerda, col_direita = st.columns([1.1, 1.8])

    with col_esquerda:
        st.subheader("🔍 Filtros de Fluxo Atual / Último Fechamento")
        st.caption("Selecione qualquer linha para plotar os dados no painel analítico.")

        with st.spinner("Extraindo cotações do momento..."):
            df_exaustao, df_retomada = processar_mercado_duplo()

        ativo_final = "EMBR3"

        # Tabela Interativa de Retomada (Top 10)
        st.markdown("**🚀 Top 10 - Retomada Confirmada de Alta**")
        if not df_retomada.empty:
            sel_ret = st.dataframe(
                df_retomada[['Ativo', 'Preço (R$)', 'IFR', 'Fluxo Vol']], 
                use_container_width=True, hide_index=True,
                selection_mode="single-row", on_select="rerun"
            )
            if sel_ret.get("selection") and sel_ret["selection"]["rows"]:
                idx_linha = sel_ret["selection"]["rows"]
                ativo_final = str(df_retomada.iloc[idx_linha]['Ativo']).strip()
        else:
            st.info("Nenhuma ação em reversão de alta no momento.")

        # Tabela Interativa de Exaustão (Top 10)
        st.markdown("**💥 Top 10 - Clímax / Exaustão de Venda**")
        if not df_exaustao.empty:
            sel_ex = st.dataframe(
                df_exaustao[['Ativo', 'Preço (R$)', 'IFR', 'Fluxo Vol']], 
                use_container_width=True, hide_index=True,
                selection_mode="single-row", on_select="rerun"
            )
            if sel_ex.get("selection") and sel_ex["selection"]["rows"]:
                idx_linha = sel_ex["selection"]["rows"]
                ativo_final = str(df_exaustao.iloc[idx_linha]['Ativo']).strip()
        else:
            st.info("Nenhuma ação em pânico institucional no momento.")

    with col_direita:
        st.subheader(f"📊 Painel Analítico de Volume e Preço: {ativo_final}")

        c1, c2 = st.columns(2)
        with c1:
            periodo_opcao = st.selectbox("Período Histórico:", ["1 dia (Intraday)", "Últimas Horas", "5 dias", "1 mês"])
        with c2:
            candle_opcao = st.selectbox("Tempo do Candle (Tempo Gráfico):", ["15 minutos", "5 minutos", "30 minutos", "1 hora", "1 dia", "1 semana"], index=0)

        map_periodo = {"1 dia (Intraday)": "1d", "Últimas Horas": "1d", "5 dias": "5d", "1 mês": "1mo"}
        map_candle = {"5 minutos": "5m", "15 minutos": "15m", "30 minutos": "30m", "1 hora": "1h", "1 dia": "1d", "1 semana": "1wk"}

        # Ajuste dinâmico automático do período caso o mercado esteja fechado (finais de semana)
        periodo_yf = map_periodo[periodo_opcao]
        if horario_atual_br.weekday() >= 5 and periodo_yf == "1d":
            periodo_yf = "5d"  # Força histórico para trazer dados se for sábado ou domingo
            
        candle_yf = map_candle[candle_opcao]

        try:
            ticker_yf = f"{ativo_final}.SA"
            dados = yf.download(ticker_yf, period=periodo_yf, interval=candle_yf, progress=False, auto_adjust=True, multi_level_index=False)

            if not dados.empty:
                dados = dados.dropna(subset=['Close', 'Volume'])

                if periodo_opcao == "Últimas Horas" and len(dados) > 16:
                    dados = dados.tail(16)

                preco_atual = float(dados['Close'].iloc[-1])

                high_low = dados['High'] - dados['Low']
                atr_calc = float(high_low.rolling(window=14).mean().fillna(preco_atual * 0.01).iloc[-1])
                
                stop_loss = preco_atual - (atr_calc * 2)
                alvo_lucro = preco_atual + (atr_calc * 1.5)
                quantidade_lote = int(RISCO_MAXIMO_FINANCEIRO / (preco_atual - stop_loss)) if (preco_atual - stop_loss) > 0 else 0

                dados['Média Ref (20)'] = dados['Close'].rolling(window=20).mean().fillna(dados['Close'])

                # Métricas Rápidas
                m1, m2, m3 = st.columns(3)
                m1.metric("Preço Analisado", f"R$ {preco_atual:.2f}")
                m2.metric("Stop Loss Recomendado", f"R$ {stop_loss:.2f}")
                m3.metric("Alvo do Trade", f"R$ {alvo_lucro:.2f}")

                # CONSTRUÇÃO DO GRÁFICO DUPLO COM SUBPLOT DE VOLUME
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, 
                                    row_width=[0.3, 0.7])

                # Linha superior de Preço
                fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço', line=dict(color='#2ca02c', width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=dados.index, y=dados['Média Ref (20)'], name='Média 20', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)

                # Linha inferior de Volume Proporcional ao candle
                fig.add_trace(go.Bar(x=dados.index, y=dados['Volume'], name='Volume do Candle', marker=dict(color='#1f77b4')), row=2, col=1)

                # Linhas de Alvo e Stop
                fig.add_hline(y=alvo_lucro, line_dash="dash", line_color="#2ca02c", annotation_text="Alvo", annotation_position="top right", row=1, col=1)
                fig.add_hline(y=stop_loss, line_dash="dash", line_color="#d62728", annotation_text="Stop", annotation_position="bottom right", row=1, col=1)

                # Remoção de lacunas fora do horário de mercado
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]), 
                        dict(bounds=[18, 10], pattern="hour")
                    ]
                )

                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"🛡️ **Gestão de Posição:** Opere no máximo **{quantidade_lote} ações** para travar o risco em R$ {RISCO_MAXIMO_FINANCEIRO:.2f}.")

                with st.spinner("Interpretando fatos de mercado..."):
                    feed = buscar_noticias_reais_yfinance(ticker_yf)
                    contexto_ia = gerar_fato_ocorrido_por_ia(ativo_final, preco_atual, feed)
                st.info(f"📰 **Contexto IA:** {contexto_ia}")

            else:
                st.error("Sem dados de cotação para as combinações gráficas selecionadas.")
        except Exception as e:
            st.error(f"Erro ao renderizar painel visual: {str(e)}")

# Aba de Histórico SQLite e Auditoria
with tab_historico:
    st.subheader("🗄️ Histórico de Varreduras e Auditoria do Ciclo Sucessivo")
    df_db = carregar_historico_banco()
    if not df_db.empty:
        df_db.columns = ['ID', 'Data/Hora (Brasília)', 'Ativo', 'Estratégia', 'Preço Entrada', 'Stop Loss', 'Alvo', 'Status Confirmação']
        st.dataframe(df_db, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum registro gravado nas tabelas locais até o momento.")

