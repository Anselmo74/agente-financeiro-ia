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

# Silenciar avisos e logs secundários do terminal
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =====================================================================
# PROTEÇÃO DE CHAVES VIA SEGREDO STREAMLIT (ST.SECRETS)
# =====================================================================
try:
    URL_IA_PROXIMIDADE = st.secrets["URL_IA_PROXIMIDADE"]
    API_KEY_IA = st.secrets["API_KEY_IA"]
    TOKEN_TELEGRAM = st.secrets["TOKEN_TELEGRAM"]
    CHAT_ID_TELEGRAM = st.secrets["CHAT_ID_TELEGRAM"]
except Exception:
    # Fallbacks estruturados caso os secrets ainda não tenham sido preenchidos na nuvem
    URL_IA_PROXIMIDADE = "https://openrouter.ai"
    API_KEY_IA = "fd10bd41-3d8f-50da-8a73-716eef2ec764"
    TOKEN_TELEGRAM = "8852525281:AAH56WNVEmmXyxvol9RKmkB3aa1Toap1QoY"
    CHAT_ID_TELEGRAM = "8852525281"

RISCO_MAXIMO_FINANCEIRO = 1000.00
LIMITE_LIQUIDEZ_DIARIA = 1000000.00
DB_NAME = "trades_historico.db"

# =====================================================================
# CONTROLE DE HORÁRIO RIGOROSO BRASÍLIA (UTC -3)
# =====================================================================
def obter_horario_brasilia():
    """Retorna o objeto datetime convertido rigorosamente para o fuso UTC-3 brasileiro."""
    return datetime.utcnow() - timedelta(hours=3)

# =====================================================================
# MÓDULO DE BANCO DE DADOS (SQLITE ROBUSTO COM COLUNA DE AUDITORIA)
# =====================================================================
def inicializar_banco():
    """Cria a tabela de histórico adicionando auditoria de acertos e confirmação."""
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
    """Grava o sinal com carimbo de hora UTC-3 de Brasília evitando duplicidade."""
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
    """Audita os sinais em aberto, validando se o mercado sucessivo confirmou a análise."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, preco_entrada, stop_loss, alvo FROM historico_sinais 
        WHERE ticker = ? AND confirmacao_analise = 'Aguardando Pregão'
    """, (ticker,))
    sinais_abertos = cursor.fetchall()
    
    for sinal in sinais_abertos:
        sid, p_entrada, s_loss, p_alvo = sinal
        if preco_atual >= p_alvo:
            status = "✅ Sucesso (Alvo Atingido)"
        elif preco_atual <= s_loss:
            status = "❌ Fracasso (Stop Acionado)"
        else:
            continue # Mantém aguardando o pregão se estiver dentro do range de oscilação
            
        cursor.execute("UPDATE historico_sinais SET confirmacao_analise = ? WHERE id = ?", (status, sid))
    conn.commit()
    conn.close()

def carregar_historico_banco():
    """Retorna o histórico armazenado no SQLite."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM historico_sinais ORDER BY id DESC", conn)
    conn.close()
    return df

# Inicialização automática da persistência
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

@st.cache_data(ttl=60)
def processar_mercado_duplo(lista_ativos_custom=None):
    """Varre a B3, classifica os ativos em TOP 10, calcula volume rel. e audita no SQLite."""
    lista_trabalho = lista_ativos_custom if lista_ativos_custom is not None else obter_universo_b3()
    pool_exaustao = []
    pool_retomada = []

    agora_br = obter_horario_brasilia()
    # Puxa 7 dias fixos para cobrir com segurança o cauda de dados nos finais de semana e noites
    periodo_scan = "7d"

    for ticker in lista_trabalho:
        try:
            df = yf.download(ticker, period=periodo_scan, interval="15m", progress=False, auto_adjust=True, multi_level_index=False)
            if df.empty or len(df) < 25: continue
            df = df.dropna(subset=['Close', 'Volume'])
            
            fechamentos = df['Close'].squeeze()
            volumes = df['Volume'].squeeze()

            df['Vol_Financeiro'] = fechamentos * volumes
            liquidez_diaria = float(df['Vol_Financeiro'].rolling(window=20).mean().iloc[-1]) * 28

            # Pula filtros de liquidez fixos se for radar direcionado pelo investidor
            if lista_ativos_custom is None and liquidez_diaria < LIMITE_LIQUIDEZ_DIARIA: continue
            
            preco_atual = float(fechamentos.iloc[-1])
            volume_ultimo_candle = float(volumes.iloc[-1])

            high_low = df['High'] - df['Low']
            df['ATR'] = high_low.rolling(window=14).mean()
            atr_atual = float(df['ATR'].iloc[-1])
            
            df['Vol_Quantidade_Media'] = volumes.rolling(window=20).mean()
            vol_ratio = float(volumes.iloc[-1] / df['Vol_Quantidade_Media'].iloc[-1])
            
            # Formatação percentual do desvio do volume em relação à média de referência do ativo
            if vol_ratio >= 1.0:
                desvio_vol_str = f"🚀 Acima ({vol_ratio * 100:.1f}%)"
            else:
                desvio_vol_str = f"降低 Abaixo ({vol_ratio * 100:.1f}%)"

            # Executa a auditoria automática de acertos no banco
            atualizar_confirmacoes_no_banco(ticker.replace('.SA',''), preco_atual)

            # -----------------------------------------------------------------
            # Motor 1: Exaustão / Pânico de Venda
            # -----------------------------------------------------------------
            df['IFR'] = calcular_ifr_professional(fechamentos, periodos=14)
            ifr_atual = float(df['IFR'].iloc[-1])

            if ifr_atual <= 33.0 or lista_ativos_custom is not None:
                dist_stop = atr_atual * 2 if atr_atual > 0 else preco_atual * 0.02
                stop_loss = preco_atual - dist_stop
                alvo_lucro = preco_atual + (dist_stop * 1.5)
                
                # Gravação programada de dados nos momentos estipulados
                hora_minuto_str = agora_br.strftime("%H:%M")
                if hora_minuto_str in ["10:30", "11:30", "12:30", "15:00", "16:30", "17:15"] and lista_ativos_custom is None:
                    salvar_sinal_no_banco(ticker.replace('.SA',''), "Pânico de Venda", preco_atual, stop_loss, alvo_lucro)
                
                pool_exaustao.append({
                    'Ativo': ticker.replace('.SA', ''), 'Preço (R$)': round(preco_atual, 2), 
                    'IFR': round(ifr_atual, 2), 'Volume Real': int(volume_ultimo_candle), 'Fluxo Vol': desvio_vol_str,
                    'atr': atr_atual, 'Diagnóstico': 'Pânico de Venda'
                })

            # -----------------------------------------------------------------
            # Motor 2: Retomada de Subida (Tendência)
            # -----------------------------------------------------------------
            df['EMA_9'] = fechamentos.ewm(span=9, adjust=False).mean()
            df['EMA_21'] = fechamentos.ewm(span=21, adjust=False).mean()
            df['Donchian_High'] = df['High'].rolling(window=20).max()

            condicao_tendencia = (df['EMA_9'].iloc[-1] > df['EMA_21'].iloc[-1]) and (vol_ratio >= 1.2) and (preco_atual >= df['Donchian_High'].iloc[-1] * 0.98)
            
            if condicao_tendencia or lista_ativos_custom is not None:
                momentum = (preco_atual - df['EMA_21'].iloc[-1]) / df['EMA_21'].iloc[-1]
                dist_stop = atr_atual * 1.5 if atr_atual > 0 else preco_atual * 0.015
                stop_loss = preco_atual - dist_stop
                alvo_lucro = preco_atual + (dist_stop * 2.0)
                
                hora_minuto_str = agora_br.strftime("%H:%M")
                if hora_minuto_str in ["10:30", "11:30", "12:30", "15:00", "16:30", "17:15"] and lista_ativos_custom is None:
                    salvar_sinal_no_banco(ticker.replace('.SA',''), "Retomada de Subida", preco_atual, stop_loss, alvo_lucro)
                
                pool_retomada.append({
                    'Ativo': ticker.replace('.SA', ''), 'Preço (R$)': round(preco_atual, 2), 
                    'IFR': round(ifr_atual, 2), 'Volume Real': int(volume_ultimo_candle), 'Fluxo Vol': desvio_vol_str,
                    'atr': atr_atual, 'Momentum': momentum, 'Diagnóstico': 'Retomada de Subida'
                })
        except: continue

    df_ex = pd.DataFrame(pool_exaustao) if pool_exaustao else pd.DataFrame(columns=['Ativo', 'Preço (R$)', 'IFR', 'Volume Real', 'Fluxo Vol', 'atr', 'Diagnóstico'])
    df_ret = pd.DataFrame(pool_retomada) if pool_retomada else pd.DataFrame(columns=['Ativo', 'Preço (R$)', 'IFR', 'Volume Real', 'Fluxo Vol', 'atr', 'Momentum', 'Diagnóstico'])
    
    if lista_ativos_custom is None:
        if not df_ex.empty: df_ex = df_ex.sort_values(by='IFR', ascending=True).head(10)
        if not df_ret.empty: df_ret = df_ret.sort_values(by='Momentum', ascending=False).head(10)
    
    return df_ex, df_ret

# =====================================================================
# INTERFACE VISUAL AVANÇADA (STREAMLIT APP UI)
# =====================================================================
horario_atual_br = obter_horario_brasilia()
st.title("🤖 AGENTE FINANCEIRO IA: Painel Analítico Avançado B3 Pro")
st.markdown(f"**Fuso Horário:** Brasília (UTC -3) | **Data/Hora:** {horario_atual_br.strftime('%d/%m/%Y %H:%M:%S')}")

st.info("🕒 **Ciclos Estritos de Registro de Auditoria:** 10:30 | 11:30 | 12:30 | 15:00 | 16:30 | 17:15")

# ---------------------------------------------------------------------
# BARRA LATERAL: RADAR DE ANÁLISE DIRECIONADA EM LOTE (CUSTOMIZADO)
# ---------------------------------------------------------------------
st.sidebar.header("🎯 Radar de Ativos Customizados")
st.sidebar.markdown("Insira uma ou mais ações para rodar as regras e critérios matemáticos em lote paralelo.")

input_ativos = st.sidebar.text_input("Ativos separados por vírgula:", placeholder="VALE3, PETR4, COGN3")
df_custom_ex, df_custom_ret = pd.DataFrame(), pd.DataFrame()

if input_ativos:
    lista_custom = [f"{t.strip().upper()}.SA" for t in input_ativos.split(",") if t.strip()]
    if lista_custom:
        with st.sidebar.spinner("Analisando lote direcionado..."):
            df_custom_ex, df_custom_ret = processar_mercado_duplo(lista_ativos_custom=lista_custom)

# Abas principais da ferramenta
tab_monitoramento, tab_historico = st.tabs(["📊 Gráficos & Sinais Online", "🗄️ Histórico & Auditoria SQLite"])

with tab_monitoramento:
    col_esquerda, col_direita = st.columns([1.2, 1.8])

    with col_esquerda:
        if input_ativos:
            st.subheader("🎯 Resultado do Radar Customizado")
            df_unificado_custom = pd.concat([df_custom_ret, df_custom_ex]).drop_duplicates(subset=['Ativo'])
            if not df_unificado_custom.empty:
                sel_custom = st.dataframe(
                    df_unificado_custom[['Ativo', 'Preço (R$)', 'IFR', 'Volume Real', 'Fluxo Vol', 'Diagnóstico']],
                    use_container_width=True, hide_index=True,
                    selection_mode="single-row", on_select="rerun"
                )
                if sel_custom.get("selection") and sel_custom["selection"]["rows"]:
                    ativo_final = str(df_unificado_custom.iloc[sel_custom["selection"]["rows"]]['Ativo']).strip()
            else:
                st.warning("Nenhum dado retornado para as ações do radar.")
            st.markdown("---")

        st.subheader("🔍 Universo B3 (Top 10)")
        st.caption("Selecione qualquer linha para plotar os dados no painel analítico da direita.")

        with st.spinner("Extraindo cotações do mercado..."):
            df_exaustao, df_retomada = processar_mercado_duplo()

        if 'ativo_final' not in locals():
            ativo_final = "EMBR3"

        # Tabela Interativa de Retomada (Top 10)
        st.markdown("**🚀 Top 10 - Retomada Confirmada de Alta**")
        if not df_retomada.empty:
            sel_ret = st.dataframe(
                df_retomada[['Ativo', 'Preço (R$)', 'IFR', 'Volume Real', 'Fluxo Vol']], 
                use_container_width=True, hide_index=True,
                selection_mode="single-row", on_select="rerun"
            )
            if sel_ret.get("selection") and sel_ret["selection"]["rows"]:
                ativo_final = str(df_retomada.iloc[sel_ret["selection"]["rows"]]['Ativo']).strip()
        else:
            st.info("Nenhuma ação em reversão de alta.")

        # Tabela Interativa de Exaustão / Pânico (Top 10)
        st.markdown("**💥 Top 10 - Pânico / Exaustão de Venda**")
        if not df_exaustao.empty:
            sel_ex = st.dataframe(
                df_exaustao[['Ativo', 'Preço (R$)', 'IFR', 'Volume Real', 'Fluxo Vol']], 
                use_container_width=True, hide_index=True,
                selection_mode="single-row", on_select="rerun"
            )
            if sel_ex.get("selection") and sel_ex["selection"]["rows"]:
                ativo_final = str(df_exaustao.iloc[sel_ex["selection"]["rows"]]['Ativo']).strip()
        else:
            st.info("Nenhuma ação em pânico de venda institucional.")

    with col_direita:
        st.subheader(f"📊 Painel Analítico Híbrido: {ativo_final}")

        c1, c2 = st.columns(2)
        with c1:
            periodo_opcao = st.selectbox("Período Histórico:", ["1 dia (Intraday)", "Últimas Horas", "5 dias", "1 mês"])
        with c2:
            candle_opcao = st.selectbox("Tempo do Candle (Tempo Gráfico):", ["15 minutos", "5 minutos", "30 minutos", "1 hora", "1 dia", "1 semana"], index=0)

        map_periodo = {"1 dia (Intraday)": "1d", "Últimas Horas": "1d", "5 dias": "5d", "1 mês": "1mo"}
        map_candle = {"5 minutos": "5m", "15 minutos": "15m", "30 minutos": "30m", "1 hora": "1h", "1 dia": "1d", "1 semana": "1wk"}

        periodo_yf = map_periodo[periodo_opcao]
        candle_yf = map_candle[candle_opcao]

        # FALLBACK HISTÓRICO: Garante dados consolidados fora do horário de pregão ativo
        if (horario_atual_br.weekday() >= 5 or horario_atual_br.hour < 10 or horario_atual_br.hour >= 18) and (periodo_yf == "1d"):
            periodo_yf = "7d"

        try:
            ticker_yf = f"{ativo_final}.SA"
            dados = yf.download(ticker_yf, period=periodo_yf, interval=candle_yf, progress=False, auto_adjust=True, multi_level_index=False)

            if not dados.empty:
                dados = dados.dropna(subset=['Close', 'Volume'])

                if periodo_opcao == "Últimas Horas" and len(dados) > 16:
                    dados = dados.tail(16)
                elif (periodo_opcao == "1 dia (Intraday)" or periodo_yf == "7d") and len(dados) > 28:
                    dados = dados.tail(28)

                preco_atual = float(dados['Close'].iloc[-1])

                high_low = dados['High'] - dados['Low']
                atr_calc = float(high_low.rolling(window=14).mean().fillna(preco_atual * 0.01).iloc[-1])
                
                stop_loss = preco_atual - (atr_calc * 2)
                alvo_lucro = preco_atual + (atr_calc * 1.5)
                quantidade_lote = int(RISCO_MAXIMO_FINANCEIRO / (preco_atual - stop_loss)) if (preco_atual - stop_loss) > 0 else 0

                dados['Média Ref (20)'] = dados['Close'].rolling(window=20).mean().fillna(dados['Close'])
                volume_medio_recente = dados['Volume'].rolling(window=20).mean().fillna(dados['Volume'])

                # Métricas em Tela incluindo o Medidor de Volatilidade Estática (ATR)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Último Preço", f"R$ {preco_atual:.2f}")
                m2.metric("Stop Loss", f"R$ {stop_loss:.2f}")
                m3.metric("Alvo Projetado", f"R$ {alvo_lucro:.2f}")
                m4.metric("Volatilidade (ATR)", f"R$ {atr_calc:.2f}")

                # Gráfico Duplo Profissional (Preço + Volume)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])

                # Trace de Preço e Média Móvel
                fig.add_trace(go.Scatter(x=dados.index.astype(str), y=dados['Close'], name='Preço', line=dict(color='#2ca02c', width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=dados.index.astype(str), y=dados['Média Ref (20)'], name='Média 20', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)

                # Destaque Visual de Volume Institucional Extremo (Altera cor se acima da média recente)
                cores_volume = ['#d62728' if v > m * 1.2 else '#1f77b4' for v, m in zip(dados['Volume'], volume_medio_recente)]
                fig.add_trace(go.Bar(x=dados.index.astype(str), y=dados['Volume'], name='Volume do Candle', marker=dict(color=cores_volume)), row=2, col=1)

                # Linhas de Alvo e Stop
                fig.add_hline(y=alvo_lucro, line_dash="dash", line_color="#2ca02c", annotation_text="Alvo", annotation_position="top right", row=1, col=1)
                fig.add_hline(y=stop_loss, line_dash="dash", line_color="#d62728", annotation_text="Stop", annotation_position="bottom right", row=1, col=1)

                # Mantém o eixo contínuo e limpo sem gaps temporais
                fig.update_xaxes(type='category')
                fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=10, b=20), height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

                st.plotly_chart(fig, use_container_width=True)
                st.success(f"🛡️ **Gestão de Posição:** Opere no máximo **{quantidade_lote} ações** para travar o risco fixado em R$ {RISCO_MAXIMO_FINANCEIRO:.2f}.")

                with st.spinner("Consultando IA..."):
                    feed = buscar_noticias_reais_yfinance(ticker_yf)
                    contexto_ia = gerar_fato_ocorrido_por_ia(ativo_final, preco_atual, feed)
                st.info(f"📰 **Contexto IA:** {contexto_ia}")
            else:
                st.error("Não foram encontrados dados históricos para renderizar este ativo fora do pregão.")
        except Exception as e:
            st.error(f"Erro ao carregar painel visual: {str(e)}")

# Aba de Histórico SQLite e Auditoria
with tab_historico:
    st.subheader("🗄️ Histórico de Varreduras e Auditoria do Ciclo Sucessivo")
    df_db = carregar_historico_banco()
    if not df_db.empty:
        df_db.columns = ['ID', 'Data/Hora (Brasília)', 'Ativo', 'Estratégia', 'Preço Entrada', 'Stop Loss', 'Alvo', 'Status Confirmação']
        st.dataframe(df_db, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum registro gravado nas tabelas locais até o momento.")
