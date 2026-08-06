import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import os
from typing import Dict, List, Tuple
import warnings
from datetime import time

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURAÇÃO PÁGINA
# =====================================================================
st.set_page_config(
    page_title="🚀 AGENTE FINANCEIRO IA PRO - 24/7",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# FUNÇÃO PARA DETECTAR MODO (PREGÃO OU HISTÓRICO)
# =====================================================================
def detectar_modo_automatico():
    """Detecta automaticamente se é horário de pregão ou não."""
    agora = datetime.now()
    hora_atual = agora.time()
    dia_semana = agora.weekday()  # 0=segunda, 4=sexta, 5=sábado, 6=domingo
    
    # Pregão B3: Segunda a Sexta, 09:30 às 17:00
    pregao_ativo = (
        dia_semana < 5 and  # Segunda a Sexta
        time(9, 30) <= hora_atual <= time(17, 0)
    )
    
    return pregao_ativo

MODO_PREGAO = detectar_modo_automatico()

# =====================================================================
# CONFIGURAÇÕES GLOBAIS
# =====================================================================
DB_HISTORICO = "analise_historica.db"
DB_TRADES = "trades_historico.db"

UNIVERSO_B3_COMPLETO = [
    "VALE3", "PETR3", "PETR4", "ITUB4", "BBDC3", "BBDC4", "BBAS3", "ITSA4", "BRFS3",
    "EMBR3", "WEGE3", "LREN3", "MGLU3", "RADL3", "PCAR3", "ELET3", "ELET6", "JBSS3",
    "ASAI3", "AZUL4", "FLRY3", "ABEV3", "CSAN3", "CMIN3", "CPFE3", "CPLE6", "ENGI11",
    "EQTL3", "EZTC3", "GGBR4", "GOAU4", "HAPV3", "HYBR3", "IRBR3", "KLBN11", "PRIO3",
    "PETZ3", "RDOR3", "RAIL3", "SBSP3", "SMTO3", "STBP3", "SUZB3", "TAEE11", "VIVT3",
    "TIMS3", "TOTS3", "TRPL4", "USIM5", "VAMO3", "VBBR3", "YDUQ3", "RENT3", "CCRO3",
    "CMIG4", "COGN3", "CRFB3", "CVCB3", "CYRE3", "DXCO3", "MULT3", "RECV3"
]

TOKEN_TELEGRAM = st.secrets.get("TOKEN_TELEGRAM", "8852525281:AAH56WNVEmmXyxvol9RKmkB3aa1Toap1QoY")
CHAT_ID_TELEGRAM = st.secrets.get("CHAT_ID_TELEGRAM", "8852525281")
API_KEY_IA = st.secrets.get("API_KEY_IA", "fd10bd41-3d8f-50da-8a73-716eef2ec764")

RISCO_MAXIMO_FINANCEIRO = 1000.00
LIMITE_LIQUIDEZ_DIARIA = 1000000.00
URL_IA_PROXIMIDADE = "https://openrouter.ai"

# =====================================================================
# GERENCIADOR DE BANCO DE DADOS
# =====================================================================
class HistoricoAnalyzer:
    """Gerenciador de análise histórica e backtests."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        """Inicializa banco de dados."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_resultado (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_backtest TEXT,
                ticker TEXT,
                data_inicio TEXT,
                data_fim TEXT,
                estrategia TEXT,
                parametros TEXT,
                total_trades INTEGER,
                trades_vencedores INTEGER,
                trades_perdedores INTEGER,
                win_rate REAL,
                payoff REAL,
                lucro_total REAL,
                lucro_medio REAL,
                prejuizo_medio REAL,
                drawdown REAL,
                roi REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades_historicos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                data_entrada TEXT,
                preco_entrada REAL,
                data_saida TEXT,
                preco_saida REAL,
                stop_loss REAL,
                alvo REAL,
                tipo_saida TEXT,
                resultado REAL,
                percentual_ganho REAL,
                duracao_dias INTEGER,
                estrategia TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analise_diaria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT,
                ticker TEXT,
                preco_abertura REAL,
                preco_fechamento REAL,
                preco_minimo REAL,
                preco_maximo REAL,
                volume REAL,
                ifr REAL,
                rsi REAL,
                vol_ratio REAL,
                liquides_financeira REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sinais_tempo_real (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                tipo_sinal TEXT,
                preco REAL,
                stop_loss REAL,
                alvo REAL,
                confianca REAL,
                ativo INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def salvar_backtest(self, resultado: Dict):
        """Salva resultado do backtest."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO backtest_resultado
            (data_backtest, ticker, data_inicio, data_fim, estrategia, parametros,
             total_trades, trades_vencedores, trades_perdedores, win_rate, payoff,
             lucro_total, lucro_medio, prejuizo_medio, drawdown, roi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            resultado['ticker'],
            resultado['data_inicio'],
            resultado['data_fim'],
            resultado['estrategia'],
            str(resultado['parametros']),
            resultado['total_trades'],
            resultado['vencedores'],
            resultado['perdedores'],
            resultado['win_rate'],
            resultado['payoff'],
            resultado['lucro_total'],
            resultado['lucro_medio'],
            resultado['prejuizo_medio'],
            resultado['drawdown'],
            resultado['roi']
        ))
        
        conn.commit()
        conn.close()
    
    def get_historico_backtests(self, ticker: str = None, dias: int = 90) -> pd.DataFrame:
        """Retorna histórico de backtests."""
        conn = sqlite3.connect(self.db_name)
        
        if ticker:
            query = f"""
                SELECT * FROM backtest_resultado 
                WHERE ticker = '{ticker}'
                AND data_backtest > datetime('now', '-{dias} days')
                ORDER BY data_backtest DESC
            """
        else:
            query = f"""
                SELECT * FROM backtest_resultado 
                WHERE data_backtest > datetime('now', '-{dias} days')
                ORDER BY data_backtest DESC
            """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

# =====================================================================
# MOTOR DE CÁLCULOS TÉCNICOS
# =====================================================================
class MotorTecnico:
    """Motor de cálculos de indicadores técnicos."""
    
    @staticmethod
    def calcular_ifr(series: pd.Series, periodos: int = 14) -> pd.Series:
        """Calcula IFR."""
        delta = series.diff()
        ganho = delta.clip(lower=0)
        perda = -delta.clip(upper=0)
        ma_ganho = ganho.ewm(alpha=1/periodos, adjust=False).mean()
        ma_perda = perda.ewm(alpha=1/periodos, adjust=False).mean()
        return 100 - (100 / (1 + (ma_ganho / ma_perda.replace(0, np.nan)))).fillna(100)
    
    @staticmethod
    def calcular_rsi(series: pd.Series, periodos: int = 14) -> pd.Series:
        """Calcula RSI."""
        delta = series.diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=periodos).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodos).mean()
        rs = ganho / perda
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calcular_atr(high: pd.Series, low: pd.Series, close: pd.Series, periodos: int = 14) -> pd.Series:
        """Calcula ATR."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(periodos).mean()

# =====================================================================
# MOTOR DE BACKTESTING
# =====================================================================
class BacktestAvancado:
    """Sistema profissional de backtesting."""
    
    @staticmethod
    def backtest_sobrevenda(ticker: str, data_inicio: str, data_fim: str, parametros: Dict) -> Dict:
        """Executa backtest da estratégia Sobrevenda + Volume."""
        try:
            df = yf.download(f"{ticker}.SA", start=data_inicio, end=data_fim, interval="1d", 
                           progress=False, auto_adjust=True, multi_level_index=False)
            
            if df.empty or len(df) < 20:
                return {'erro': 'Dados insuficientes'}
            
            df['IFR'] = MotorTecnico.calcular_ifr(df['Close'])
            df['RSI'] = MotorTecnico.calcular_rsi(df['Close'])
            df['Vol_Media'] = df['Volume'].rolling(window=20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_Media']
            df['ATR'] = MotorTecnico.calcular_atr(df['High'], df['Low'], df['Close'])
            
            trades = []
            em_posicao = False
            preco_entrada = 0
            data_entrada = None
            stop_loss = 0
            alvo = 0
            
            for idx in range(1, len(df)):
                data_atual = df.index[idx].strftime("%Y-%m-%d")
                preco = df['Close'].iloc[idx]
                ifr = df['IFR'].iloc[idx]
                vol_ratio = df['Vol_Ratio'].iloc[idx]
                atr = df['ATR'].iloc[idx]
                
                if not em_posicao and ifr < parametros['ifr_limite'] and vol_ratio > parametros['vol_ratio_min']:
                    em_posicao = True
                    preco_entrada = preco
                    data_entrada = data_atual
                    stop_loss = preco_entrada - (atr * parametros['atr_multiplicador_stop'])
                    alvo = preco_entrada + (atr * parametros['atr_multiplicador_alvo'])
                
                elif em_posicao:
                    tipo_saida = None
                    preco_saida = None
                    
                    if preco <= stop_loss:
                        tipo_saida = 'STOP_LOSS'
                        preco_saida = stop_loss
                    elif preco >= alvo:
                        tipo_saida = 'ALVO_ATINGIDO'
                        preco_saida = alvo
                    
                    if tipo_saida:
                        resultado = preco_saida - preco_entrada
                        pct = (resultado / preco_entrada) * 100
                        dias = (pd.to_datetime(data_atual) - pd.to_datetime(data_entrada)).days
                        
                        trades.append({
                            'data_entrada': data_entrada,
                            'preco_entrada': preco_entrada,
                            'data_saida': data_atual,
                            'preco_saida': preco_saida,
                            'stop_loss': stop_loss,
                            'alvo': alvo,
                            'tipo_saida': tipo_saida,
                            'resultado': resultado,
                            'pct': pct,
                            'dias': dias
                        })
                        
                        em_posicao = False
            
            if trades:
                trades_df = pd.DataFrame(trades)
                vencedores = len(trades_df[trades_df['resultado'] > 0])
                perdedores = len(trades_df[trades_df['resultado'] < 0])
                win_rate = (vencedores / len(trades_df)) * 100 if len(trades_df) > 0 else 0
                
                lucro_medio = trades_df[trades_df['resultado'] > 0]['resultado'].mean() if vencedores > 0 else 0
                prejuizo_medio = abs(trades_df[trades_df['resultado'] < 0]['resultado'].mean()) if perdedores > 0 else 1
                payoff = lucro_medio / prejuizo_medio if prejuizo_medio > 0 else 0
                
                lucro_total = trades_df['resultado'].sum()
                drawdown = ((trades_df['resultado'].cumsum().min() / abs(trades_df['resultado'].sum() if trades_df['resultado'].sum() != 0 else 1)) * 100) if trades_df['resultado'].sum() != 0 else 0
                roi = (lucro_total / (abs(prejuizo_medio) * perdedores)) * 100 if perdedores > 0 else 100
                
                return {
                    'ticker': ticker,
                    'data_inicio': data_inicio,
                    'data_fim': data_fim,
                    'estrategia': 'Sobrevenda + Volume',
                    'parametros': parametros,
                    'total_trades': len(trades_df),
                    'vencedores': vencedores,
                    'perdedores': perdedores,
                    'win_rate': win_rate,
                    'payoff': payoff,
                    'lucro_total': lucro_total,
                    'lucro_medio': lucro_medio,
                    'prejuizo_medio': prejuizo_medio,
                    'drawdown': drawdown,
                    'roi': roi,
                    'trades': trades_df
                }
            else:
                return {'erro': 'Nenhum trade gerado'}
        
        except Exception as e:
            return {'erro': str(e)}

# =====================================================================
# ANÁLISE DE ERROS
# =====================================================================
class AnalisadorErros:
    """Analisa padrões de erros e acertos."""
    
    @staticmethod
    def analisar_trades(trades_df: pd.DataFrame) -> Dict:
        """Analisa características dos trades vencedores vs perdedores."""
        vencedores = trades_df[trades_df['resultado'] > 0]
        perdedores = trades_df[trades_df['resultado'] < 0]
        
        analise = {
            'vencedores': {
                'count': len(vencedores),
                'ganho_medio': vencedores['resultado'].mean() if len(vencedores) > 0 else 0,
                'ganho_maximo': vencedores['resultado'].max() if len(vencedores) > 0 else 0,
                'ganho_minimo': vencedores['resultado'].min() if len(vencedores) > 0 else 0,
                'pct_medio': vencedores['pct'].mean() if len(vencedores) > 0 else 0,
                'duracao_media': vencedores['dias'].mean() if len(vencedores) > 0 else 0
            },
            'perdedores': {
                'count': len(perdedores),
                'perda_media': perdedores['resultado'].mean() if len(perdedores) > 0 else 0,
                'perda_maxima': perdedores['resultado'].max() if len(perdedores) > 0 else 0,
                'perda_minima': perdedores['resultado'].min() if len(perdedores) > 0 else 0,
                'pct_medio': perdedores['pct'].mean() if len(perdedores) > 0 else 0,
                'duracao_media': perdedores['dias'].mean() if len(perdedores) > 0 else 0
            }
        }
        
        return analise

# =====================================================================
# INTERFACE PRINCIPAL
# =====================================================================

# Banner Superior com Status
st.markdown("---")
col_status_1, col_status_2, col_status_3 = st.columns([1, 3, 1])

with col_status_1:
    if MODO_PREGAO:
        st.info("🟢 PREGÃO ATIVO")
    else:
        st.warning("🔴 PREGÃO FECHADO")

with col_status_2:
    agora = datetime.now()
    st.write(f"⏰ {agora.strftime('%d/%m/%Y %H:%M:%S')}")

with col_status_3:
    st.write(f"📊 {len(UNIVERSO_B3_COMPLETO)} Ativos")

st.markdown("---")

st.title("🚀 AGENTE FINANCEIRO IA PRO - 24/7")
st.markdown("Sistema Inteligente que Funciona em Tempo Real + Análise Histórica Completa")

# Sidebar
st.sidebar.title("⚙️ MODO DE OPERAÇÃO")

if MODO_PREGAO:
    st.sidebar.success("🟢 PREGÃO ATIVO - Modo Tempo Real")
    modo_selecionado = st.sidebar.radio(
        "Escolha o módulo:",
        ["📡 Scanner Tempo Real", "📊 Dashboard Gráfico", "🎯 Gerenciador de Posições",
         "🔔 Alertas Personalizados", "📈 Performance Diária"]
    )
else:
    st.sidebar.warning("🔴 PREGÃO FECHADO - Modo Histórico")
    modo_selecionado = st.sidebar.radio(
        "Escolha o módulo:",
        ["🔍 Análise Histórica", "🧪 Backtest Detalhado", "📈 Comparativo Estratégias",
         "❌ Análise de Erros", "✅ Melhores Operações", "📊 Relatório Executivo"]
    )

analisador_historico = HistoricoAnalyzer(DB_HISTORICO)

# =====================================================================
# MODO PREGÃO - TEMPO REAL
# =====================================================================
if MODO_PREGAO:
    
    if modo_selecionado == "📡 Scanner Tempo Real":
        st.header("📡 SCANNER TEMPO REAL - Sinais Agora")
        
        st.info("""
        🔥 Scanner em tempo real buscando:
        ✓ IFR < 30 (Sobrevenda)
        ✓ Volume Ratio > 1.5 (Volume Anormal)
        ✓ Liquidez > R$ 1M/dia
        """)
        
        if st.button("🔎 ESCANEAR AGORA"):
            with st.spinner("Escaneando 60 ativos..."):
                sinais = []
                
                for ticker in UNIVERSO_B3_COMPLETO:
                    try:
                        df = yf.download(f"{ticker}.SA", period="100d", interval="1d", 
                                       progress=False, auto_adjust=True, multi_level_index=False)
                        
                        if len(df) < 20:
                            continue
                        
                        df['IFR'] = MotorTecnico.calcular_ifr(df['Close'])
                        df['Vol_Media'] = df['Volume'].rolling(20).mean()
                        df['Vol_Ratio'] = df['Volume'] / df['Vol_Media']
                        df['ATR'] = MotorTecnico.calcular_atr(df['High'], df['Low'], df['Close'])
                        
                        ifr_atual = df['IFR'].iloc[-1]
                        vol_ratio_atual = df['Vol_Ratio'].iloc[-1]
                        preco_atual = df['Close'].iloc[-1]
                        atr_atual = df['ATR'].iloc[-1]
                        
                        if ifr_atual < 30 and vol_ratio_atual > 1.5:
                            stop_loss = preco_atual - (atr_atual * 2.0)
                            alvo = preco_atual + (atr_atual * 3.0)
                            
                            sinais.append({
                                'Ticker': ticker,
                                'Preço': f"R$ {preco_atual:.2f}",
                                'IFR': f"{ifr_atual:.1f}",
                                'Vol Ratio': f"{vol_ratio_atual:.2f}x",
                                'Stop Loss': f"R$ {stop_loss:.2f}",
                                'Alvo': f"R$ {alvo:.2f}",
                                'Confiança': f"{(ifr_atual / 30 * 100):.0f}%" if ifr_atual < 30 else "Baixa"
                            })
                    except:
                        pass
                
                if sinais:
                    df_sinais = pd.DataFrame(sinais)
                    st.success(f"✅ {len(sinais)} Sinais Encontrados!")
                    st.dataframe(df_sinais, use_container_width=True)
                else:
                    st.warning("❌ Nenhum sinal identificado no momento")
    
    elif modo_selecionado == "📊 Dashboard Gráfico":
        st.header("📊 DASHBOARD GRÁFICO - Acompanhamento Tempo Real")
        
        ticker_dashboard = st.selectbox("Escolha o ativo", UNIVERSO_B3_COMPLETO)
        intervalo = st.selectbox("Intervalo", ["1d", "1h", "15m", "5m"])
        
        if st.button("📥 Carregar Gráfico"):
            with st.spinner("Carregando dados..."):
                df = yf.download(f"{ticker_dashboard}.SA", period="60d", interval=intervalo,
                               progress=False, auto_adjust=True, multi_level_index=False)
                
                df['IFR'] = MotorTecnico.calcular_ifr(df['Close'])
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                           low=df['Low'], close=df['Close'], name='Candles'))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')))
                
                fig.update_layout(title=f"{ticker_dashboard} - {intervalo}", template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Preço Atual", f"R$ {df['Close'].iloc[-1]:.2f}")
                col_m2.metric("IFR", f"{df['IFR'].iloc[-1]:.1f}")
                col_m3.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
                col_m4.metric("Variação", f"{((df['Close'].iloc[-1] - df['Close'].iloc[-50]) / df['Close'].iloc[-50] * 100):.2f}%")
    
    elif modo_selecionado == "🎯 Gerenciador de Posições":
        st.header("🎯 GERENCIADOR DE POSIÇÕES - Controle em Tempo Real")
        
        st.info("Gerencie suas posições abertas, defina stops e alvos")
        
        col_pos1, col_pos2 = st.columns(2)
        
        with col_pos1:
            ticker_pos = st.selectbox("Ativo", UNIVERSO_B3_COMPLETO)
            preco_entrada = st.number_input("Preço Entrada (R$)", 1.0, 1000.0, 100.0)
        
        with col_pos2:
            qtd_acoes = st.number_input("Quantidade", 1, 10000, 100)
            tipo_posicao = st.radio("Tipo", ["Compra", "Venda"])
        
        col_pos3, col_pos4 = st.columns(2)
        
        with col_pos3:
            stop_loss = st.number_input("Stop Loss (R$)", 1.0, 1000.0, 90.0)
        
        with col_pos4:
            alvo_lucro = st.number_input("Alvo Lucro (R$)", 1.0, 1000.0, 110.0)
        
        if st.button("📊 CALCULAR RISCO/RETORNO"):
            risco = abs(preco_entrada - stop_loss) * qtd_acoes
            retorno = abs(alvo_lucro - preco_entrada) * qtd_acoes
            razao_risco_retorno = retorno / risco if risco > 0 else 0
            
            col_calc1, col_calc2, col_calc3 = st.columns(3)
            col_calc1.metric("Risco (R$)", f"R$ {risco:.2f}", f"-{(abs(preco_entrada - stop_loss) / preco_entrada * 100):.2f}%")
            col_calc2.metric("Retorno (R$)", f"R$ {retorno:.2f}", f"+{(abs(alvo_lucro - preco_entrada) / preco_entrada * 100):.2f}%")
            col_calc3.metric("Razão RR", f"{razao_risco_retorno:.2f}:1", "Bom" if razao_risco_retorno > 2 else "Ruim")

# =====================================================================
# MODO HISTÓRICO - ANÁLISE COMPLETA
# =====================================================================
else:
    
    if modo_selecionado == "🔍 Análise Histórica":
        st.header("🔍 ANÁLISE DE HISTÓRICO - Visualize o Passado")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker_select = st.selectbox("Escolha o ativo", UNIVERSO_B3_COMPLETO)
        
        with col2:
            periodo_dias = st.selectbox("Período", [30, 60, 90, 180, 365], index=2)
        
        with col3:
            tipo_analise = st.selectbox("Tipo", ["Diário", "Semanal", "Mensal"])
        
        if st.button("📥 Carregar Histórico"):
            with st.spinner("Carregando dados históricos..."):
                data_fim = datetime.now()
                data_inicio = data_fim - timedelta(days=periodo_dias)
                
                try:
                    df = yf.download(
                        f"{ticker_select}.SA",
                        start=data_inicio.strftime("%Y-%m-%d"),
                        end=data_fim.strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                        multi_level_index=False
                    )
                    
                    if not df.empty:
                        df['IFR'] = MotorTecnico.calcular_ifr(df['Close'])
                        df['RSI'] = MotorTecnico.calcular_rsi(df['Close'])
                        df['Vol_Media'] = df['Volume'].rolling(20).mean()
                        df['Vol_Ratio'] = df['Volume'] / df['Vol_Media']
                        df['SMA_20'] = df['Close'].rolling(20).mean()
                        df['SMA_50'] = df['Close'].rolling(50).mean()
                        
                        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                        
                        col_metric1.metric(
                            "Preço Atual",
                            f"R$ {df['Close'].iloc[-1]:.2f}",
                            f"{((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):.2f}%"
                        )
                        col_metric2.metric("Máxima", f"R$ {df['High'].max():.2f}")
                        col_metric3.metric("Mínima", f"R$ {df['Low'].min():.2f}")
                        col_metric4.metric("Vol Médio", f"{df['Volume'].mean():,.0f}")
                        
                        st.markdown("---")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Fechamento', line=dict(color='#2ca02c')))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='#ff7f0e', dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='#d62728', dash='dash')))
                        
                        fig.update_layout(title=f"Histórico {ticker_select} - {periodo_dias} dias", template="plotly_dark", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        fig_ifr = px.area(x=df.index, y=df['IFR'], title="Índice de Força Relativa (IFR)", template="plotly_dark")
                        fig_ifr.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Sobrevenda")
                        fig_ifr.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Sobrecompra")
                        st.plotly_chart(fig_ifr, use_container_width=True)
                        
                        st.subheader("📊 Dados Históricos")
                        df_exibir = df[['Close', 'Volume', 'IFR', 'RSI', 'Vol_Ratio']].tail(20)
                        st.dataframe(df_exibir, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {str(e)}")
    
    elif modo_selecionado == "🧪 Backtest Detalhado":
        st.header("🧪 BACKTEST DETALHADO - Teste Estratégias no Passado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ticker_backtest = st.selectbox("Ativo para Backtest", UNIVERSO_B3_COMPLETO, key="backtest_ticker")
        
        with col2:
            periodo_backtest = st.selectbox("Período", [30, 60, 90, 180, 365], index=3)
        
        st.subheader("⚙️ Parâmetros da Estratégia")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        
        with col_p1:
            ifr_limite = st.slider("Limite IFR (Sobrevenda)", 10, 40, 30)
        
        with col_p2:
            vol_ratio_min = st.slider("Vol Ratio Mínimo", 1.0, 3.0, 1.5, 0.1)
        
        with col_p3:
            atr_stop = st.slider("ATR Stop Loss", 1.0, 3.0, 2.0, 0.1)
        
        with col_p4:
            atr_alvo = st.slider("ATR Alvo", 2.0, 5.0, 3.0, 0.1)
        
        if st.button("▶️ EXECUTAR BACKTEST"):
            with st.spinner("Processando backtest histórico..."):
                data_fim = datetime.now()
                data_inicio = data_fim - timedelta(days=periodo_backtest)
                
                parametros = {
                    'ifr_limite': ifr_limite,
                    'vol_ratio_min': vol_ratio_min,
                    'atr_multiplicador_stop': atr_stop,
                    'atr_multiplicador_alvo': atr_alvo
                }
                
                resultado = BacktestAvancado.backtest_sobrevenda(
                    ticker_backtest,
                    data_inicio.strftime("%Y-%m-%d"),
                    data_fim.strftime("%Y-%m-%d"),
                    parametros
                )
                
                if 'erro' not in resultado:
                    analisador_historico.salvar_backtest(resultado)
                    
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    
                    col_res1.metric("Total de Trades", resultado['total_trades'])
                    col_res2.metric("Win Rate", f"{resultado['win_rate']:.1f}%")
                    col_res3.metric("Payoff", f"{resultado['payoff']:.2f}x")
                    col_res4.metric("Lucro Total", f"R$ {resultado['lucro_total']:.2f}")
                    
                    st.markdown("---")
                    
                    trades_df = resultado['trades']
                    trades_df['Equity'] = trades_df['resultado'].cumsum()
                    
                    fig_equity = px.line(trades_df, x=trades_df.index, y='Equity',
                                        title='Curva de Equity', template="plotly_dark")
                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                    fig_dist = px.histogram(trades_df, x='pct', nbins=20,
                                          title='Distribuição de Ganho/Perda (%)', template="plotly_dark")
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.subheader("📋 Detalhamento de Trades")
                    st.dataframe(trades_df, use_container_width=True)
                
                else:
                    st.error(f"Erro no backtest: {resultado['erro']}")
    
    elif modo_selecionado == "❌ Análise de Erros":
        st.header("❌ ANÁLISE DE ERROS - Aprenda com os Fracassos")
        
        ticker_erro = st.selectbox("Escolha o ativo", UNIVERSO_B3_COMPLETO, key="erro_ticker")
        periodo_erro = st.selectbox("Período (dias)", [30, 60, 90, 180], key="erro_periodo")
        
        if st.button("🔍 ANALISAR ERROS"):
            with st.spinner("Analisando padrões de erro..."):
                data_fim = datetime.now()
                data_inicio = data_fim - timedelta(days=periodo_erro)
                
                parametros = {
                    'ifr_limite': 30,
                    'vol_ratio_min': 1.5,
                    'atr_multiplicador_stop': 2.0,
                    'atr_multiplicador_alvo': 3.0
                }
                
                resultado = BacktestAvancado.backtest_sobrevenda(
                    ticker_erro,
                    data_inicio.strftime("%Y-%m-%d"),
                    data_fim.strftime("%Y-%m-%d"),
                    parametros
                )
                
                if 'erro' not in resultado:
                    trades_df = resultado['trades']
                    analise = AnalisadorErros.analisar_trades(trades_df)
                    
                    col_v, col_p = st.columns(2)
                    
                    with col_v:
                        st.subheader("✅ TRADES VENCEDORES")
                        st.metric("Quantidade", analise['vencedores']['count'])
                        st.metric("Ganho Médio", f"R$ {analise['vencedores']['ganho_medio']:.2f}")
                        st.metric("Ganho Máximo", f"R$ {analise['vencedores']['ganho_maximo']:.2f}")
                        st.metric("% Médio", f"{analise['vencedores']['pct_medio']:.2f}%")
                    
                    with col_p:
                        st.subheader("❌ TRADES PERDEDORES")
                        st.metric("Quantidade", analise['perdedores']['count'])
                        st.metric("Perda Média", f"R$ {analise['perdedores']['perda_media']:.2f}")
                        st.metric("Perda Máxima", f"R$ {analise['perdedores']['perda_maxima']:.2f}")
                        st.metric("% Médio", f"{analise['perdedores']['pct_medio']:.2f}%")
                
                else:
                    st.error(f"Erro: {resultado['erro']}")
    
    elif modo_selecionado == "✅ Melhores Operações":
        st.header("✅ MELHORES OPERAÇÕES - Analise Seus Sucessos")
        
        ticker_best = st.selectbox("Escolha o ativo", UNIVERSO_B3_COMPLETO, key="best_ticker")
        periodo_best = st.selectbox("Período (dias)", [30, 60, 90, 180, 365], key="best_periodo")
        
        if st.button("🏆 ANALISAR MELHORES"):
            with st.spinner("Buscando melhores operações..."):
                data_fim = datetime.now()
                data_inicio = data_fim - timedelta(days=periodo_best)
                
                parametros = {
                    'ifr_limite': 30,
                    'vol_ratio_min': 1.5,
                    'atr_multiplicador_stop': 2.0,
                    'atr_multiplicador_alvo': 3.0
                }
                
                resultado = BacktestAvancado.backtest_sobrevenda(
                    ticker_best,
                    data_inicio.strftime("%Y-%m-%d"),
                    data_fim.strftime("%Y-%m-%d"),
                    parametros
                )
                
                if 'erro' not in resultado:
                    trades_df = resultado['trades']
                    melhores = trades_df.nlargest(5, 'resultado')
                    
                    st.subheader("🏆 Top 5 Melhores Trades")
                    
                    for idx, (i, trade) in enumerate(melhores.iterrows(), 1):
                        with st.expander(f"#{idx} - {trade['data_entrada']} | Ganho: R$ {trade['resultado']:.2f}"):
                            col_best1, col_best2, col_best3, col_best4 = st.columns(4)
                            col_best1.metric("Entrada", f"R$ {trade['preco_entrada']:.2f}")
                            col_best2.metric("Saída", f"R$ {trade['preco_saida']:.2f}")
                            col_best3.metric("Lucro", f"R$ {trade['resultado']:.2f}")
                            col_best4.metric("Duração", f"{trade['dias']} dias")
                else:
                    st.error(f"Erro: {resultado['erro']}")
    
    elif modo_selecionado == "📊 Relatório Executivo":
        st.header("📊 RELATÓRIO EXECUTIVO - Resumo Completo")
        
        df_hist = analisador_historico.get_historico_backtests(dias=90)
        
        if not df_hist.empty:
            col_agg1, col_agg2, col_agg3, col_agg4 = st.columns(4)
            
            col_agg1.metric("Total de Backtests", len(df_hist))
            col_agg2.metric("Win Rate Médio", f"{df_hist['win_rate'].mean():.1f}%")
            col_agg3.metric("Payoff Médio", f"{df_hist['payoff'].mean():.2f}x")
            col_agg4.metric("ROI Médio", f"{df_hist['roi'].mean():.1f}%")
            
            st.markdown("---")
            st.dataframe(df_hist[['data_backtest', 'ticker', 'estrategia', 'total_trades', 'win_rate', 'payoff']], use_container_width=True)
        
        else:
            st.warning("Nenhum backtest encontrado")

# =====================================================================
# RODAPÉ
# =====================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
🤖 AGENTE FINANCEIRO IA PRO v2.0 | Modo Automático: {} | Última Atualização: {}
</div>
""".format(
    "🟢 TEMPO REAL" if MODO_PREGAO else "🔴 HISTÓRICO",
    datetime.now().strftime("%H:%M:%S")
), unsafe_allow_html=True)
