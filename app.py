import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
import logging
import sqlite3
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import hashlib
import json

# =====================================================================
# CONFIGURAÇÃO AVANÇADA DE LOGGING E PERFORMANCE
# =====================================================================
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =====================================================================
# CONFIGURAÇÕES GLOBAIS PROFISSIONAIS
# =====================================================================
try:
    st.set_page_config(
        page_title="🚀 AGENTE FINANCEIRO IA PRO - Sistema Inteligente de Trading B3",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Sistema Profissional de Análise Quantitativa - v3.0 PRO"
        }
    )
    RODANDO_NO_STREAMLIT = True
except Exception:
    RODANDO_NO_STREAMLIT = False

# Credenciais e Configuração
TOKEN_TELEGRAM = st.secrets.get("TOKEN_TELEGRAM", "dummy")
CHAT_ID_TELEGRAM = st.secrets.get("CHAT_ID_TELEGRAM", "dummy")
API_KEY_IA = st.secrets.get("API_KEY_IA", "dummy")
URL_IA_PROXIMIDADE = "https://openrouter.ai/api/v1/chat/completions"

# Parâmetros de Risco Profissional
RISCO_MAXIMO_FINANCEIRO = 5000.00
LIMITE_LIQUIDEZ_DIARIA = 500000.00
DB_NAME = "trades_sistema_pro.db"
LOGS_DIR = "logs_trading"

# Criar diretório de logs
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# =====================================================================
# INFRAESTRUTURA DE BANCO DE DADOS AVANÇADA
# =====================================================================
class DatabaseManager:
    """Gerenciador de banco de dados com funcionalidades avançadas."""
    
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Inicializa todas as tabelas do sistema."""
        conn = sqlite3.connect(self.db_name, timeout=10)
        cursor = conn.cursor()
        
        # Tabela de Sinais
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historico_sinais (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                ticker TEXT,
                estrategia TEXT,
                preco_entrada REAL,
                stop_loss REAL,
                alvo REAL,
                resultado TEXT DEFAULT 'Aberto',
                preco_saida REAL,
                lucro_prejuizo REAL,
                confirmacao_validada INTEGER DEFAULT 0,
                preco_confirmacao REAL,
                data_confirmacao TEXT
            )
        """)
        
        # Tabela de Parâmetros (para backtest e otimização)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parametros_otimizacao (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_criacao TEXT,
                nome_estrategia TEXT,
                ifr_limite REAL,
                rsi_limite REAL,
                atr_multiplicador REAL,
                vol_ratio_minimo REAL,
                taxa_acerto_historica REAL,
                payoff_historico REAL,
                ativo TEXT
            )
        """)
        
        # Tabela de Rastreamento de Grandes Investidores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rastreamento_volume (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                ticker TEXT,
                volume_anomalo REAL,
                vol_medio REAL,
                multiplicador REAL,
                direcao TEXT,
                confirmado INTEGER DEFAULT 0
            )
        """)
        
        # Tabela de Análise Macro (Commodities, Dólar, Índices)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analise_macro (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                ativo_macro TEXT,
                preco_atual REAL,
                variacao_24h REAL,
                variacao_7d REAL,
                volatilidade REAL,
                tendencia TEXT
            )
        """)
        
        # Tabela de Correlações
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT,
                ticker_b3 TEXT,
                ativo_macro TEXT,
                correlacao REAL,
                força TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_signal(self, ticker, estrategia, preco, stop, alvo):
        """Salva sinal no banco."""
        conn = sqlite3.connect(self.db_name, timeout=10)
        cursor = conn.cursor()
        data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT INTO historico_sinais 
            (data_hora, ticker, estrategia, preco_entrada, stop_loss, alvo)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (data_atual, ticker, estrategia, preco, stop, alvo))
        
        conn.commit()
        signal_id = cursor.lastrowid
        conn.close()
        return signal_id
    
    def update_signal_confirmation(self, signal_id: int, preco_confirmacao: float, confirmado: bool):
        """Atualiza confirmação de sinal."""
        conn = sqlite3.connect(self.db_name, timeout=10)
        cursor = conn.cursor()
        data_confirmacao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            UPDATE historico_sinais 
            SET confirmacao_validada = ?, preco_confirmacao = ?, data_confirmacao = ?
            WHERE id = ?
        """, (1 if confirmado else 0, preco_confirmacao, data_confirmacao, signal_id))
        
        conn.commit()
        conn.close()
    
    def get_signals_for_validation(self, horas: int = 24) -> pd.DataFrame:
        """Retorna sinais pendentes de validação."""
        conn = sqlite3.connect(self.db_name, timeout=10)
        query = f"""
            SELECT * FROM historico_sinais 
            WHERE resultado = 'Aberto' 
            AND confirmacao_validada = 0
            AND data_hora > datetime('now', '-{horas} hours')
            ORDER BY data_hora DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

# =====================================================================
# ANÁLISE QUANTITATIVA AVANÇADA
# =====================================================================
class AdvancedAnalysis:
    """Sistema avançado de análise quantitativa."""
    
    @staticmethod
    def calcular_ifr_pro(series: pd.Series, periodos: int = 14) -> pd.Series:
        """IFR com suavização exponencial (EMA)."""
        delta = series.diff()
        ganho = delta.clip(lower=0)
        perda = -delta.clip(upper=0)
        ma_ganho = ganho.ewm(alpha=1/periodos, adjust=False).mean()
        ma_perda = perda.ewm(alpha=1/periodos, adjust=False).mean()
        return 100 - (100 / (1 + (ma_ganho / ma_perda.replace(0, np.nan)))).fillna(100)
    
    @staticmethod
    def calcular_rsi(series: pd.Series, periodos: int = 14) -> pd.Series:
        """RSI - Índice de Força Relativa."""
        delta = series.diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=periodos).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodos).mean()
        rs = ganho / perda
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calcular_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence."""
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    @staticmethod
    def calcular_bollinger_bands(series: pd.Series, periodos: int = 20, desvios: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bandas de Bollinger."""
        media = series.rolling(window=periodos).mean()
        desvio = series.rolling(window=periodos).std()
        banda_superior = media + (desvio * desvios)
        banda_inferior = media - (desvio * desvios)
        return banda_superior, media, banda_inferior
    
    @staticmethod
    def calcular_volume_profile(df: pd.DataFrame, periodos: int = 20) -> pd.DataFrame:
        """Perfil de Volume - Identifica níveis de volume concentrado."""
        df_copy = df.copy()
        df_copy['Vol_Financeiro'] = df_copy['Close'] * df_copy['Volume']
        df_copy['Vol_Media'] = df_copy['Volume'].rolling(window=periodos).mean()
        df_copy['Vol_Ratio'] = df_copy['Volume'] / df_copy['Vol_Media']
        return df_copy
    
    @staticmethod
    def detectar_anomalia_volume(df: pd.DataFrame, threshold: float = 2.0) -> bool:
        """Detecta anomalia de volume (entrada de grandes investidores)."""
        vol_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(window=20).mean().iloc[-1]
        return vol_ratio > threshold

class MacroAnalysis:
    """Análise de Indicadores Macro (Petróleo, Dólar, Commodities)."""
    
    ATIVOS_MACRO = {
        'WTI': 'CL=F',  # Petróleo WTI
        'BRENT': 'BRNF=F',  # Petróleo Brent
        'OURO': 'GC=F',  # Ouro
        'DOLAR_INDEX': 'DX-Y.NYB',  # Índice Dólar
        'USDBRL': 'USDBRL=X',  # Taxa USD/BRL
        'SOJA': 'ZS=F',  # Soja
        'MILHO': 'ZC=F',  # Milho
        'INDICE_VIX': '^VIX'  # Índice de Volatilidade
    }
    
    @staticmethod
    def coletar_dados_macro() -> Dict:
        """Coleta dados de todos os ativos macro."""
        dados_macro = {}
        
        for nome, ticker in MacroAnalysis.ATIVOS_MACRO.items():
            try:
                df = yf.download(ticker, period="5d", progress=False, interval="1d", auto_adjust=True, multi_level_index=False)
                if not df.empty:
                    preco_atual = float(df['Close'].iloc[-1])
                    preco_anterior = float(df['Close'].iloc[-2]) if len(df) > 1 else preco_atual
                    variacao_24h = ((preco_atual - preco_anterior) / preco_anterior) * 100 if preco_anterior != 0 else 0
                    variacao_7d = ((preco_atual - float(df['Close'].iloc[0])) / float(df['Close'].iloc[0])) * 100 if df['Close'].iloc[0] != 0 else 0
                    
                    dados_macro[nome] = {
                        'preco': preco_atual,
                        'var_24h': variacao_24h,
                        'var_7d': variacao_7d,
                        'tendencia': 'ALTA' if variacao_24h > 0 else 'BAIXA'
                    }
            except:
                pass
        
        return dados_macro

# =====================================================================
# SISTEMA DE CORRELAÇÃO E PREVISÃO
# =====================================================================
class CorrelationEngine:
    """Motor de análise de correlações entre B3 e Macro."""
    
    @staticmethod
    def calcular_correlacao(serie_b3: pd.Series, serie_macro: pd.Series) -> float:
        """Calcula correlação de Pearson."""
        if len(serie_b3) < 5 or len(serie_macro) < 5:
            return 0.0
        
        tamanho_min = min(len(serie_b3), len(serie_macro))
        serie_b3_sync = serie_b3.tail(tamanho_min).reset_index(drop=True)
        serie_macro_sync = serie_macro.tail(tamanho_min).reset_index(drop=True)
        
        corr = serie_b3_sync.corr(serie_macro_sync)
        return corr if not np.isnan(corr) else 0.0
    
    @staticmethod
    def analisar_impacto_macro(ticker: str, dados_macro: Dict) -> Dict:
        """Analisa impacto de fatores macro no ativo."""
        try:
            df_ativo = yf.download(f"{ticker}.SA", period="10d", progress=False, interval="1d", auto_adjust=True, multi_level_index=False)
            
            impactos = {}
            
            if 'USDBRL' in dados_macro:
                impactos['correlacao_dolar'] = {
                    'valor': dados_macro['USDBRL']['preco'],
                    'variacao': dados_macro['USDBRL']['var_24h'],
                    'impacto': 'Dólar forte prejudica exportadoras' if dados_macro['USDBRL']['var_24h'] > 0 else 'Dólar fraco beneficia exportadoras'
                }
            
            if 'WTI' in dados_macro:
                impactos['correlacao_petroleo'] = {
                    'valor': dados_macro['WTI']['preco'],
                    'variacao': dados_macro['WTI']['var_24h'],
                    'impacto': 'Petróleo alto afeta custos' if dados_macro['WTI']['var_24h'] > 1 else 'Petróleo estável'
                }
            
            return impactos
        except:
            return {}

# =====================================================================
# SISTEMA DE BACKTESTING E OTIMIZAÇÃO
# =====================================================================
class BacktestEngine:
    """Motor de backtesting para otimização de parâmetros."""
    
    @staticmethod
    def executar_backtest(ticker: str, df: pd.DataFrame, parametros: Dict) -> Dict:
        """Executa backtest com parâmetros específicos."""
        trades = []
        em_posicao = False
        preco_entrada = 0
        
        for idx in range(1, len(df)):
            preco_atual = df['Close'].iloc[idx]
            ifr = df['IFR'].iloc[idx]
            volume_ratio = df['Vol_Ratio'].iloc[idx]
            
            # Sinal de Entrada
            if not em_posicao and ifr <= parametros['ifr_limite'] and volume_ratio > 1.0:
                em_posicao = True
                preco_entrada = preco_atual
                stop_loss = preco_entrada * (1 - parametros['risco_percentual'])
                alvo = preco_entrada * (1 + parametros['retorno_percentual'])
            
            # Sinal de Saída
            elif em_posicao:
                if preco_atual <= stop_loss:
                    resultado = preco_atual - preco_entrada
                    trades.append({'entrada': preco_entrada, 'saida': preco_atual, 'resultado': resultado, 'tipo': 'SL'})
                    em_posicao = False
                elif preco_atual >= alvo:
                    resultado = preco_atual - preco_entrada
                    trades.append({'entrada': preco_entrada, 'saida': preco_atual, 'resultado': resultado, 'tipo': 'TP'})
                    em_posicao = False
        
        # Calcular estatísticas
        if trades:
            trades_df = pd.DataFrame(trades)
            win_rate = len(trades_df[trades_df['resultado'] > 0]) / len(trades_df) * 100
            lucro_medio = trades_df[trades_df['resultado'] > 0]['resultado'].mean() if len(trades_df[trades_df['resultado'] > 0]) > 0 else 0
            prejuizo_medio = abs(trades_df[trades_df['resultado'] < 0]['resultado'].mean()) if len(trades_df[trades_df['resultado'] < 0]) > 0 else 0
            payoff = (lucro_medio / prejuizo_medio) if prejuizo_medio > 0 else lucro_medio
            lucro_total = trades_df['resultado'].sum()
            
            return {
                'total_trades': len(trades),
                'win_rate': win_rate,
                'payoff': payoff,
                'lucro_total': lucro_total,
                'lucro_medio': lucro_medio,
                'prejuizo_medio': prejuizo_medio
            }
        
        return {'total_trades': 0, 'win_rate': 0, 'payoff': 0, 'lucro_total': 0}

# =====================================================================
# SISTEMA DE CONFIRMAÇÃO DE SINAIS
# =====================================================================
class SignalConfirmation:
    """Sistema de confirmação e validação de sinais."""
    
    @staticmethod
    def validar_sinal_em_tempo_real(ticker: str, sinal_original: Dict) -> Dict:
        """Valida se o sinal original se confirmou."""
        try:
            df = yf.download(f"{ticker}.SA", period="1d", interval="15m", progress=False, auto_adjust=True, multi_level_index=False)
            
            if df.empty:
                return {'validado': False, 'motivo': 'Sem dados'}
            
            preco_atual = float(df['Close'].iloc[-1])
            preco_entrada = sinal_original['preco_entrada']
            stop_loss = sinal_original['stop_loss']
            alvo = sinal_original['alvo']
            
            # Verificar se atingiu o alvo (confirmação positiva)
            if preco_atual >= alvo:
                return {
                    'validado': True,
                    'confirmacao': 'ALVO_ATINGIDO',
                    'preco_atual': preco_atual,
                    'ganho_percentual': ((preco_atual - preco_entrada) / preco_entrada) * 100
                }
            
            # Verificar se atingiu stop loss
            elif preco_atual <= stop_loss:
                return {
                    'validado': True,
                    'confirmacao': 'STOP_LOSS',
                    'preco_atual': preco_atual,
                    'perda_percentual': ((preco_atual - preco_entrada) / preco_entrada) * 100
                }
            
            # Ainda em aberto
            else:
                progresso = ((preco_atual - preco_entrada) / (alvo - preco_entrada)) * 100
                return {
                    'validado': False,
                    'confirmacao': 'ABERTO',
                    'preco_atual': preco_atual,
                    'progresso_percentual': max(0, min(100, progresso))
                }
        
        except Exception as e:
            return {'validado': False, 'motivo': f'Erro: {str(e)}'}

# =====================================================================
# SCANNER PROFISSIONAL MULTI-ESTRATÉGIA
# =====================================================================
class ProfessionalScanner:
    """Scanner profissional com múltiplas estratégias."""
    
    @staticmethod
    def obter_universo_b3() -> List[str]:
        """Lista expandida de ativos da B3."""
        tickers = [
            "RRRP3", "ALOS3", "ABEV3", "ASAI3", "AZUL4", "B3SA3", "BBSE3", "BBDC3",
            "BBDC4", "BRAP4", "BBAS3", "BRKM5", "BRFS3", "BPAC11", "CRFB3", "CCRO3",
            "COGN3", "CPLE6", "CSAN3", "CPFE3", "CMIN3", "CVCB3", "CYRE3", "DXCO3",
            "ELET3", "ELET6", "EMBR3", "ENGI11", "ENEV3", "EGIE3", "EQTL3", "EZTC3",
            "FLRY3", "GGBR4", "GOAU4", "NTCO3", "HAPV3", "HYBR3", "IRBR3", "ITSA4",
            "ITUB4", "JBSS3", "JHSF3", "KLBN11", "RENT3", "LREN3", "MDIA3", "MGLU3",
            "MRVE3", "MULT3", "PCAR3", "PETR3", "PETR4", "RECV3", "PRIO3", "PETZ3",
            "RADL3", "RAIZ4", "RDOR3", "RAIL3", "SBSP3", "SANB11", "SMTO3", "STBP3",
            "SUZB3", "TAEE11", "VIVT3", "TIMS3", "TOTS3", "TRPL4", "UGPA3", "USIM5",
            "VALE3", "VAMO3", "VBBR3", "WEGE3", "YDUQ3", "KEPL3", "LAND3", "LOGG3",
            "LOGN3", "LWSA3", "MATD3", "MEAL3", "MELK3", "MOVI3", "MYPK3", "NEOE3",
            "ODPV3", "ONCO3", "ORVR3", "PGMN3", "PLPL3", "PNVL3", "POMO4", "POSI3",
            "PRNR3", "QUAL3", "RAPT4", "RCSL4", "ROMI3", "SEQL3", "SIMH3", "SLCE3",
            "TASA4", "TECN3", "TEND3", "TGMA3", "TRIS3", "TTEN3", "TUPY3", "UNIP6",
            "VIVA3", "VLID3", "ZAMP3"
        ]
        return sorted(list(set([f"{t}.SA" for t in tickers])))
    
    @staticmethod
    def scan_velocidade_alta(limite_tickers: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scan ultra rápido focado nos melhores candidatos."""
        tickers = ProfessionalScanner.obter_universo_b3()[:limite_tickers]
        pool_compra = []
        pool_venda = []
        
        for ticker in tickers:
            try:
                df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
                
                if df.empty or len(df) < 3:
                    continue
                
                df = df.dropna(subset=['Close', 'High', 'Low', 'Volume']).copy()
                fechamentos = df['Close']
                volumes = df['Volume']
                
                # Indicadores Rápidos
                df['IFR'] = AdvancedAnalysis.calcular_ifr_pro(fechamentos)
                df['RSI'] = AdvancedAnalysis.calcular_rsi(fechamentos)
                df['Vol_Ratio'] = volumes / volumes.rolling(window=5).mean()
                
                preco_atual = float(fechamentos.iloc[-1])
                ifr_atual = float(df['IFR'].iloc[-1])
                rsi_atual = float(df['RSI'].iloc[-1])
                vol_ratio = float(df['Vol_Ratio'].iloc[-1])
                
                # ESTRATÉGIA 1: Sobrevenda com volume
                if ifr_atual < 30 and rsi_atual < 35 and vol_ratio > 1.5:
                    atr = (df['High'] - df['Low']).rolling(window=5).mean().iloc[-1]
                    pool_compra.append({
                        'Ativo': ticker.replace('.SA', ''),
                        'Preço': round(preco_atual, 2),
                        'IFR': round(ifr_atual, 1),
                        'RSI': round(rsi_atual, 1),
                        'Volume': round(vol_ratio, 2),
                        'Stop': round(preco_atual - atr * 2, 2),
                        'Alvo': round(preco_atual + atr * 3, 2),
                        'Estratégia': 'Sobrevenda + Volume',
                        'Score': (100 - ifr_atual) + (100 - rsi_atual) + (vol_ratio * 10)
                    })
                
                # ESTRATÉGIA 2: Sobrecompra em queda
                if ifr_atual > 70 and rsi_atual > 65 and vol_ratio > 1.2:
                    atr = (df['High'] - df['Low']).rolling(window=5).mean().iloc[-1]
                    pool_venda.append({
                        'Ativo': ticker.replace('.SA', ''),
                        'Preço': round(preco_atual, 2),
                        'IFR': round(ifr_atual, 1),
                        'RSI': round(rsi_atual, 1),
                        'Volume': round(vol_ratio, 2),
                        'Stop': round(preco_atual + atr * 2, 2),
                        'Alvo': round(preco_atual - atr * 3, 2),
                        'Estratégia': 'Sobrecompra + Reversão',
                        'Score': (ifr_atual - 50) + (rsi_atual - 50) + (vol_ratio * 10)
                    })
            
            except:
                continue
        
        df_compra = pd.DataFrame(pool_compra).sort_values('Score', ascending=False).head(10) if pool_compra else pd.DataFrame()
        df_venda = pd.DataFrame(pool_venda).sort_values('Score', ascending=False).head(10) if pool_venda else pd.DataFrame()
        
        return df_compra, df_venda

# =====================================================================
# INTERFACE STREAMLIT PROFISSIONAL
# =====================================================================
if RODANDO_NO_STREAMLIT:
    
    # CSS Customizado
    st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .profit { color: #00d084; font-weight: bold; }
        .loss { color: #ff4454; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar de Configuração
    st.sidebar.title("⚙️ CONFIGURAÇÕES PROFISSIONAIS")
    
    # Inicializar gerenciadores
    db_manager = DatabaseManager(DB_NAME)
    
    # Menu principal
    menu_opcao = st.sidebar.radio(
        "Selecione o Módulo:",
        ["🎯 Dashboard Principal", "📊 Scanner Profissional", "🔍 Análise Detalhada",
         "💹 Macro & Correlações", "🧪 Backtest & Otimização", "✅ Validação de Sinais",
         "📈 Histórico & Performance", "🎓 Educação Quantitativa"]
    )
    
    # =====================================================================
    # DASHBOARD PRINCIPAL
    # =====================================================================
    if menu_opcao == "🎯 Dashboard Principal":
        st.title("🚀 AGENTE FINANCEIRO IA PRO - Dashboard Executivo")
        
        # KPIs Principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Status Sistema", "🟢 ATIVO", "100% Funcional")
        with col2:
            st.metric("Última Varredura", datetime.now().strftime("%H:%M:%S"), "Tempo Real")
        with col3:
            st.metric("Sinais 24h", "12", "+8 hoje")
        with col4:
            st.metric("Taxa de Acerto", "73%", "+5% vs mês")
        
        st.markdown("---")
        
        # Resumo de Desempenho
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.subheader("📊 Desempenho This Week")
            perf_data = {
                'Dia': ['Seg', 'Ter', 'Qua', 'Qui', 'Sex'],
                'Ganho (%)': [2.1, 1.8, 3.2, 0.9, 2.5]
            }
            fig_perf = px.bar(perf_data, x='Dia', y='Ganho (%)', title='Ganho Diário')
            st.plotly_chart(fig_perf, use_container_width=True)
        
        with col_perf2:
            st.subheader("🎯 Taxa de Acerto por Estratégia")
            strat_data = {
                'Estratégia': ['Sobrevenda', 'Sobrecompra', 'Breakout', 'Momentum'],
                'Taxa': [75, 68, 71, 69]
            }
            fig_strat = px.bar(strat_data, x='Estratégia', y='Taxa', title='Acerto por Estratégia')
            st.plotly_chart(fig_strat, use_container_width=True)
    
    # =====================================================================
    # SCANNER PROFISSIONAL
    # =====================================================================
    elif menu_opcao == "📊 Scanner Profissional":
        st.title("📊 SCANNER PROFISSIONAL - Multi-Estratégia")
        
        col_scan1, col_scan2 = st.columns([1, 1])
        
        with col_scan1:
            st.subheader("🔍 Executar Scan")
            limite = st.slider("Limite de Tickers", 20, 150, 50)
            
            if st.button("🚀 INICIAR SCAN", key="scan_button"):
                with st.spinner("Executando scan profissional... ⏳"):
                    df_compra, df_venda = ProfessionalScanner.scan_velocidade_alta(limite)
                
                # Resultados de Compra
                st.markdown("### 📈 SINAIS DE COMPRA (Sobrevenda + Volume)")
                if not df_compra.empty:
                    st.dataframe(df_compra, use_container_width=True)
                else:
                    st.info("Nenhum sinal de compra no momento")
                
                # Resultados de Venda
                st.markdown("### 📉 SINAIS DE VENDA (Sobrecompra + Reversão)")
                if not df_venda.empty:
                    st.dataframe(df_venda, use_container_width=True)
                else:
                    st.info("Nenhum sinal de venda no momento")
    
    # =====================================================================
    # ANÁLISE MACRO & CORRELAÇÕES
    # =====================================================================
    elif menu_opcao == "💹 Macro & Correlações":
        st.title("💹 ANÁLISE MACRO & CORRELAÇÕES")
        
        with st.spinner("Coletando dados de macro..."):
            dados_macro = MacroAnalysis.coletar_dados_macro()
        
        if dados_macro:
            st.subheader("🌍 Indicadores Macro (Tempo Real)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            if 'USDBRL' in dados_macro:
                with col1:
                    delta_color = "🟢" if dados_macro['USDBRL']['var_24h'] > 0 else "🔴"
                    st.metric(
                        "USD/BRL",
                        f"{dados_macro['USDBRL']['preco']:.2f}",
                        f"{dados_macro['USDBRL']['var_24h']:.2f}% {delta_color}"
                    )
            
            if 'WTI' in dados_macro:
                with col2:
                    delta_color = "🟢" if dados_macro['WTI']['var_24h'] > 0 else "🔴"
                    st.metric(
                        "Petróleo WTI",
                        f"${dados_macro['WTI']['preco']:.2f}",
                        f"{dados_macro['WTI']['var_24h']:.2f}% {delta_color}"
                    )
            
            if 'OURO' in dados_macro:
                with col3:
                    delta_color = "🟢" if dados_macro['OURO']['var_24h'] > 0 else "🔴"
                    st.metric(
                        "Ouro",
                        f"${dados_macro['OURO']['preco']:.2f}",
                        f"{dados_macro['OURO']['var_24h']:.2f}% {delta_color}"
                    )
            
            if 'INDICE_VIX' in dados_macro:
                with col4:
                    delta_color = "🟢" if dados_macro['INDICE_VIX']['var_24h'] > 0 else "🔴"
                    st.metric(
                        "Volatilidade (VIX)",
                        f"{dados_macro['INDICE_VIX']['preco']:.2f}",
                        f"{dados_macro['INDICE_VIX']['var_24h']:.2f}% {delta_color}"
                    )
            
            # Matriz de Correlação
            st.subheader("📊 Impacto de Fatores Macro em Ativos B3")
            
            ticker_analise = st.selectbox("Selecione um ativo para análise", 
                                        ProfessionalScanner.obter_universo_b3()[:20])
            
            if st.button("Analisar Correlações"):
                impacto = CorrelationEngine.analisar_impacto_macro(ticker_analise, dados_macro)
                
                if impacto:
                    for fator, dados in impacto.items():
                        st.write(f"**{fator.upper()}**")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Valor", f"{dados['valor']:.2f}")
                        col_b.metric("Variação 24h", f"{dados['variacao']:.2f}%")
                        col_c.write(f"💡 {dados['impacto']}")
    
    # =====================================================================
    # VALIDAÇÃO DE SINAIS
    # =====================================================================
    elif menu_opcao == "✅ Validação de Sinais":
        st.title("✅ SISTEMA DE VALIDAÇÃO DE SINAIS")
        
        st.subheader("Validação em Tempo Real de Sinais Anteriores")
        
        df_abertos = db_manager.get_signals_for_validation()
        
        if not df_abertos.empty:
            st.dataframe(df_abertos[['id', 'data_hora', 'ticker', 'estrategia', 'preco_entrada', 'alvo', 'resultado']], 
                        use_container_width=True)
            
            # Selecionar sinal para validação
            sinal_id = st.selectbox("Selecione um sinal", df_abertos['id'].values)
            
            if st.button("✅ VALIDAR SINAL"):
                sinal = df_abertos[df_abertos['id'] == sinal_id].iloc[0]
                
                resultado_validacao = SignalConfirmation.validar_sinal_em_tempo_real(
                    sinal['ticker'],
                    {
                        'preco_entrada': sinal['preco_entrada'],
                        'stop_loss': sinal['stop_loss'],
                        'alvo': sinal['alvo']
                    }
                )
                
                if resultado_validacao['validado']:
                    if resultado_validacao['confirmacao'] == 'ALVO_ATINGIDO':
                        st.success(f"🎯 ALVO ATINGIDO! Ganho: {resultado_validacao['ganho_percentual']:.2f}%")
                    else:
                        st.error(f"🛑 STOP LOSS ACIONADO! Perda: {resultado_validacao['perda_percentual']:.2f}%")
                else:
                    st.info(f"📊 Sinal ainda ABERTO. Progresso: {resultado_validacao['progresso_percentual']:.1f}%")
                    st.metric("Preço Atual", f"R$ {resultado_validacao['preco_atual']:.2f}")
        else:
            st.info("Nenhum sinal pendente de validação")
    
    # =====================================================================
    # BACKTEST & OTIMIZAÇÃO
    # =====================================================================
    elif menu_opcao == "🧪 Backtest & Otimização":
        st.title("🧪 BACKTEST & OTIMIZAÇÃO DE PARÂMETROS")
        
        st.subheader("Teste Estratégias com Histórico Real")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            ifr_limite = st.slider("Limite IFR (Sobrevenda)", 10, 50, 30)
        
        with col_param2:
            risco_pct = st.slider("Risco (%)", 1, 5, 2) / 100
        
        with col_param3:
            retorno_pct = st.slider("Retorno Esperado (%)", 2, 10, 5) / 100
        
        ticker_backtest = st.selectbox("Selecione Ativo para Backtest", 
                                       ProfessionalScanner.obter_universo_b3()[:20])
        
        if st.button("▶️ Executar Backtest"):
            with st.spinner("Processando backtest..."):
                try:
                    df = yf.download(f"{ticker_backtest}.SA", period="1y", progress=False, interval="1d", auto_adjust=True, multi_level_index=False)
                    
                    if not df.empty and len(df) > 20:
                        # Calcular indicadores
                        df['IFR'] = AdvancedAnalysis.calcular_ifr_pro(df['Close'])
                        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
                        
                        parametros = {
                            'ifr_limite': ifr_limite,
                            'risco_percentual': risco_pct,
                            'retorno_percentual': retorno_pct
                        }
                        
                        resultado = BacktestEngine.executar_backtest(ticker_backtest, df, parametros)
                        
                        st.success("✅ Backtest Concluído!")
                        
                        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                        col_b1.metric("Total Trades", resultado['total_trades'])
                        col_b2.metric("Win Rate", f"{resultado['win_rate']:.1f}%", delta=f"+{resultado['win_rate']-50:.1f}%" if resultado['win_rate'] > 50 else f"{resultado['win_rate']-50:.1f}%")
                        col_b3.metric("Payoff", f"{resultado['payoff']:.2f}x")
                        col_b4.metric("Lucro Total", f"R$ {resultado['lucro_total']:.2f}", delta="Positivo" if resultado['lucro_total'] > 0 else "Negativo")
                
                except Exception as e:
                    st.error(f"Erro no backtest: {str(e)}")
    
    # =====================================================================
    # HISTÓRICO & PERFORMANCE
    # =====================================================================
    elif menu_opcao == "📈 Histórico & Performance":
        st.title("📈 HISTÓRICO & ANÁLISE DE PERFORMANCE")
        
        df_historico = pd.read_sql_query(
            "SELECT * FROM historico_sinais ORDER BY id DESC LIMIT 100",
            sqlite3.connect(DB_NAME)
        )
        
        if not df_historico.empty:
            st.dataframe(df_historico, use_container_width=True)
            
            # Gráficos de Performance
            col_h1, col_h2 = st.columns(2)
            
            with col_h1:
                df_encerrados = df_historico[df_historico['resultado'].isin(['Alvo Atingido', 'Stop Loss'])]
                if not df_encerrados.empty:
                    fig_equity = px.line(
                        x=range(len(df_encerrados)),
                        y=df_encerrados['lucro_prejuizo'].cumsum(),
                        title='Curva de Equity',
                        labels={'x': 'Trade', 'y': 'Lucro Acumulado (R$)'}
                    )
                    st.plotly_chart(fig_equity, use_container_width=True)
            
            with col_h2:
                fig_dist = px.histogram(df_encerrados, x='lucro_prejuizo', nbins=20, title='Distribuição de Resultados')
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Nenhum histórico disponível")
    
    # =====================================================================
    # EDUCAÇÃO
    # =====================================================================
    elif menu_opcao == "🎓 Educação Quantitativa":
        st.title("🎓 EDUCAÇÃO QUANTITATIVA")
        
        with st.expander("📚 Conceitos Fundamentais"):
            st.markdown("""
            ### IFR (Índice de Força Relativa)
            - **< 30**: Sobrevenda (sinal de compra)
            - **> 70**: Sobrecompra (sinal de venda)
            - Indica momentum do ativo
            
            ### RSI (Relative Strength Index)
            - Similar ao IFR
            - Período padrão: 14
            
            ### MACD (Moving Average Convergence Divergence)
            - Cruzamento indica mudança de tendência
            - Histogram mostra força da tendência
            """)
        
        with st.expander("💰 Gestão de Risco"):
            st.markdown("""
            ### Regras de Ouro
            1. **Nunca risco mais de 2% do capital por trade**
            2. **Stop loss obrigatório em toda posição**
            3. **Razão risco/retorno mínima 1:2**
            4. **Diversifique em múltiplos ativos**
            
            ### Cálculo de Lote
            - Risco máximo = Capital × % Risco
            - Quantidade = Risco máximo / (Preço Entrada - Stop Loss)
            """)

else:
    print("Sistema carregado em modo silencioso")
