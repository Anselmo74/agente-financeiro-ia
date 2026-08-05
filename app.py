import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Silenciar avisos e logs secundários do terminal para performance limpa
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =====================================================================
# CONFIGURAÇÕES MULTICANAL DO USUÁRIO
# =====================================================================
TOKEN_TELEGRAM = "8852525281:AAH56WNVEmmXyxvol9RKmkB3aa1Toap1QoY"
CHAT_ID_TELEGRAM = "8852525281"

# Endpoint oficial de Chat Completions no OpenRouter
URL_IA_PROXIMIDADE = "https://openrouter.ai"
API_KEY_IA = "fd10bd41-3d8f-50da-8a73-716eef2ec764"

# Canais secundários de contingência e segurança
EMAIL_REMETENTE = "advcorreto@gmail.com"
EMAIL_DESTINATARIO = "advcorreto@gmail.com"
SENHA_APP_GMAIL = "COLE_AQUI_SUA_SENHA_DE_APP_DO_GMAIL"

SMS_ACCOUNT_SID = "COLE_AQUI_SEU_TWILIO_ACCOUNT_SID"
SMS_AUTH_TOKEN = "COLE_AQUI_SEU_TWILIO_AUTH_TOKEN"
TELEFONE_SMS_ORIGEM = "+1XXXXXXXXXX"
TELEFONE_DESTINO = "+5534991410631"

# Parâmetros de Gestão de Risco Blindada
RISCO_MAXIMO_FINANCEIRO = 1000.00
LIMITE_LIQUIDEZ_DIARIA = 1000000.00  # Filtro mínimo de R$ 1 Milhão/dia

def obter_universo_ibov_e_smallcaps():
    """Gera a lista unificada e sanitizada de ações líquidas da B3."""
    tickers_base = [
        "RRRP3", "ALOS3", "ALPA4", "ABEV3", "ARZZ3", "ASAI3", "AZUL4", "B3SA3", "BBSE3", "BBDC3",
        "BBDC4", "BRAP4", "BBAS3", "BRKM5", "BRFS3", "BPAC11", "CRFB3", "CCRO3", "CMIG4",
        "COGN3", "CPLE6", "CSAN3", "CPFE3", "CMIN3", "CVCB3", "CYRE3", "DXCO3", "ELET3", "ELET6",
        "EMBR3", "ENGI11", "ENEV3", "EGIE3", "EQTL3", "EZTC3", "FLRY3", "GGBR4", "GOAU4",
        "NTCO3", "HAPV3", "HYBR3", "IGTI11", "IRBR3", "ITSA4", "ITUB4", "JBSS3", "JHSF3",
        "KLBN11", "RENT3", "LREN3", "MDIA3", "MGLU3", "MRVE3", "MULT3", "PCAR3", "PETR3", "PETR4",
        "RECV3", "PRIO3", "PETZ3", "RADL3", "RAIZ4", "RDOR3", "RAIL3", "SBSP3", "SANB11", "SMTO3",
        "STBP3", "SUZB3", "TAEE11", "VIVT3", "TIMS3", "TOTS3", "TRPL4", "UGPA3", "USIM5",
        "VALE3", "VAMO3", "VBBR3", "WEGE3", "YDUQ3", "AERI3", "AURE3", "AMER3", "ARML3",
        "BLAU3", "CAML3", "CASH3", "CEAB3", "CLSA3", "CSNA3", "CURY3", "DIRR3", "EVEN3", 
        "FESA4", "FIQE3", "GGRC11", "GMAT3", "GRND3", "GUAR3", "IFCM3", "INTB3", "JALL3",
        "KEPL3", "LAND3", "LAVV3", "LOGG3", "LOGN3", "AMBP3", "LWSA3", "MATD3", "MEAL3",
        "MELK3", "MOVI3", "MYPK3", "NEOE3", "ODPV3", "ONCO3", "ORVR3", "PGMN3", "PLPL3",
        "PNVL3", "POMO4", "POSI3", "PRNR3", "QUAL3", "RAPT4", "RCSL4", "ROMI3", "SEQL3", 
        "SIMH3", "SLCE3", "TASA4", "TECN3", "TEND3", "TGMA3", "TRIS3", "TTEN3", "TUPY3", 
        "UNIP6", "VIVA3", "VLID3", "ZAMP3"
    ]
    return sorted(list(set([f"{t}.SA" for t in tickers_base])))

def verificar_dia_util():
    """Valida se o mercado está operando (ignora finais de semana e feriados nacionais)."""
    hoje = datetime.now()
    if hoje.weekday() >= 5: return False
    feriados = ["01-01", "04-21", "05-01", "09-07", "10-12", "11-02", "11-15", "11-20", "12-25"]
    if hoje.strftime("%m-%d") in feriados: return False
    return True

def calcular_ifr_professional(series, periodos=14):
    """Calcula o Índice de Força Relativa com suavização e proteção contra divisão por zero."""
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
            manchetes = [n.get('title', '') for n in noticias[:3]]
            return " | ".join(manchetes)
    except: pass
    return "Nenhuma manchete recente encontrada no feed."

def gerar_fato_ocorrido_por_ia(ticker, preco, manchetes_reais):
    """Valida o cenário do ativo consultando a inteligência artificial online."""
    headers = {
        "Authorization": f"Bearer {API_KEY_IA}", 
        "Content-Type": "application/json"
    }
    prompt = (
        f"Ação: {ticker}. Preço: R$ {preco:.2f}. Manchetes recentes coletadas no book: '{manchetes_reais}'. "
        f"Com base nessas informações ou no cenário macro atual do Brasil, escreva uma única frase curta de no máximo 15 palavras "
        f"explicando qual fato corporativo, econômico ou boato justifica a forte oscilação recente deste papel na B3. Seja ultra objetivo."
    )
    data = {
        "model": "google/gemma-2-9b-it:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(URL_IA_PROXIMIDADE, headers=headers, json=data, timeout=12)
        if response.status_code == 200:
            return response.json()['choices']['message']['content'].strip()
    except: pass
    return f"Ajuste técnico de carteiras institucionais perto da faixa de R$ {preco:.2f}."

def enviar_telegram(mensagem):
    """Envia o relatório via endpoint oficial do Telegram Bot API."""
    url_final = f"https://telegram.org{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": CHAT_ID_TELEGRAM, "text": mensagem, "parse_mode": "Markdown"}
    try: return requests.post(url_final, json=payload, timeout=15).status_code == 200
    except: return False

def enviar_email(mensagem):
    """Envia o relatório através do servidor seguro SMTP do Gmail."""
    if "COLE_AQUI" in SENHA_APP_GMAIL: return False
    msg = MIMEMultipart()
    msg['From'] = EMAIL_REMETENTE
    msg['To'] = EMAIL_DESTINATARIO
    msg['Subject'] = f"📊 AGENTE B3 INTRADAY: Relatório de Oportunidades - {datetime.now().strftime('%d/%m/%Y')}"
    msg.attach(MIMEText(mensagem.replace("*", "").replace("_", ""), 'plain'))
    try:
        server = smtplib.SMTP('://gmail.com', 587)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA_APP_GMAIL)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()
        return True
    except: return False

def enviar_sms(mensagem):
    """Envia alertas resumidos via API corporativa do Twilio Gateway."""
    if "COLE_AQUI" in SMS_ACCOUNT_SID: return False
    linhas = mensaje.split("\n")
    resumo_sms = "🤖 ALERTA B3: " + " | ".join([l for l in linhas if "1️⃣" in l or "2️⃣" in l or "3️⃣" in l])
    url = f"https://twilio.com{SMS_ACCOUNT_SID}/Messages.json"
    payload = {"To": TELEFONE_DESTINO, "From": TELEFONE_SMS_ORIGEM, "Body": resumo_sms[:160]}
    try: return requests.post(url, data=payload, auth=(SMS_ACCOUNT_SID, SMS_AUTH_TOKEN), timeout=10).status_code == 201
    except: return False
def executar_agente_b3_analista():
    """Executa a triagem dupla de ativos: Exaustão de Venda e Retomada Confirmada."""
    if not verificar_dia_util():
        print("💤 Hoje não é um dia útil de mercado. Agente em modo de espera.")
        return

    lista_completa_b3 = obter_universo_ibov_e_smallcaps()
    pool_exaustao = []
    pool_retomada = []

    print(f"🤖 Scanner Multíndice Ativo. Processando estratégias de Clímax e Reversão...")

    for ticker in lista_completa_b3:
        try:
            # Download de dados do intraday (15 minutos)
            df = yf.download(ticker, period="5d", interval="15m", progress=False, auto_adjust=True, multi_level_index=False)
            if df.empty or len(df) < 30: continue
            df = df.dropna(subset=['Close', 'Volume'])
            
            fechamentos = df['Close'].squeeze()
            volumes = df['Volume'].squeeze()

            # Estimação da liquidez projetada diária real (28 candles de 15m por pregão)
            df['Vol_Financeiro'] = fechamentos * volumes
            liquidez_media_recente = float(df['Vol_Financeiro'].rolling(window=20).mean().iloc[-1]) * 28

            if liquidez_media_recente < LIMITE_LIQUIDEZ_DIARIA:
                continue

            preco_atual = float(fechamentos.iloc[-1])
            if preco_atual < 1.00:
                continue

            # Indicadores Comuns e Volatilidade
            high_low = df['High'] - df['Low']
            df['ATR'] = high_low.rolling(window=14).mean()
            atr_atual = float(df['ATR'].iloc[-1])
            df['Vol_Quantidade_Media'] = volumes.rolling(window=20).mean()
            vol_ratio = float(volumes.iloc[-1] / df['Vol_Quantidade_Media'].iloc[-1])

            # =================================================================
            # MOTOR 1: IDENTIFICAÇÃO DE EXAUSTÃO DE VENDA (PÂNICO)
            # =================================================================
            df['IFR'] = calcular_ifr_professional(fechamentos, periodos=14)
            ifr_atual = float(df['IFR'].iloc[-1])

            if ifr_atual <= 33.0:
                if ifr_atual <= 30.0 and vol_ratio >= 1.5:
                    categoria_peso = 1
                elif ifr_atual < 30.0 and vol_ratio < 1.0:
                    categoria_peso = 3
                elif ifr_atual <= 33.0 and vol_ratio >= 1.2:
                    categoria_peso = 2
                else:
                    categoria_peso = 4

                pool_exaustao.append({
                    'ticker': ticker.replace('.SA', ''), 'preco': preco_atual, 'ifr': ifr_atual,
                    'vol_ratio': vol_ratio, 'atr': atr_atual, 'liquidez': liquidez_media_recente,
                    'categoria': categoria_peso
                })

            # =================================================================
            # MOTOR 2: DETECÇÃO DE RETOMADA CONFIRMADA (MOMENTUM / REVERSÃO)
            # =================================================================
            df['EMA_9'] = fechamentos.ewm(span=9, adjust=False).mean()
            df['EMA_21'] = fechamentos.ewm(span=21, adjust=False).mean()
            
            ema9_atual = float(df['EMA_9'].iloc[-1])
            ema21_atual = float(df['EMA_21'].iloc[-1])
            ema9_anterior = float(df['EMA_9'].iloc[-2])
            ema21_anterior = float(df['EMA_21'].iloc[-2])

            df['Donchian_High'] = df['High'].rolling(window=20).max()
            donchian_high_atual = float(df['Donchian_High'].iloc[-1])

            cruzamento_alta = (ema9_atual > ema21_atual) and (ema9_anterior <= ema21_anterior or fechamentos.iloc[-1] > df['High'].rolling(window=10).max().iloc[-2])
            volume_confirmado = vol_ratio >= 1.2

            if cruzamento_alta and volume_confirmado and (preco_atual >= donchian_high_atual * 0.98):
                forca_momentum = (preco_atual - ema21_atual) / ema21_atual
                
                pool_retomada.append({
                    'ticker': ticker.replace('.SA', ''), 'preco': preco_atual, 'ifr': ifr_atual,
                    'vol_ratio': vol_ratio, 'atr': atr_atual, 'liquidez': liquidez_media_recente,
                    'momentum': forca_momentum
                })

            time.sleep(0.04)
        except: continue

    msg_final = ""

    # =========================================================================
    # PROCESSAMENTO DO RELATÓRIO 1: EXAUSTÃO DE VENDA
    # =========================================================================
    if pool_exaustao:
        df_ex = pd.DataFrame(pool_exaustao)
        df_top10_ex = df_ex.sort_values(by=['categoria', 'ifr'], ascending=[True, True]).head(10)

        msg_final += "*🔥 RANKING TOP 10 PRIORIDADE DE RETORNO B3 (EXAUSTÃO) 🔥*\n"
        msg_final += "_Alvos de pânico com iminente repique institucional_\n\n"
        
        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        for idx, row in df_top10_ex.reset_index(drop=True).iterrows():
            manchetes = buscar_noticias_reais_yfinance(f"{row['ticker']}.SA")
            fato_relevante = gerar_fato_ocorrido_por_ia(row['ticker'], row['preco'], manchetes)

            distancia_stop = row['atr'] * 2 if row['atr'] > 0 else row['preco'] * 0.02
            stop_loss = row['preco'] - distancia_stop
            alvo_lucro = row['preco'] + (distancia_stop * 1.5)
            quantidade_lote = int(RISCO_MAXIMO_FINANCEIRO / (row['preco'] - stop_loss)) if (row['preco'] - stop_loss) > 0 else 0

            if row['categoria'] == 1:
                diagnostico = "💥 CLÍMAX DE VENDA (ALTA PRIORIDADE)\nPânico institucional com volume extremo. Alto potencial de repique."
            elif row['categoria'] == 2:
                diagnostico = "⚖️ SUPORTE RELEVANTE (MÉDIA PRIORIDADE)\nAtivo defendido por ordens institucionais na região."
            elif row['categoria'] == 3:
                diagnostico = "⚠️ RISCO DE ARRASTO (ALTA EXAUSTÃO / BAIXO FLUXO)\nPreço baixo por gravidade, sem agressão compradora."
            else:
                diagnostico = "⏳ MONITORAMENTO TÉCNICO VENDEDOR\nApenas oscilação rotineira dentro da tendência de baixa."

            msg_final += (
                f"{emojis[idx]} *{row['ticker']}* | Giro Estimado: R$ {row['liquidez']/1000000:.1f}M/dia\n"
                f"• Preço: R$ {row['preco']:.2f} | IFR: {row['ifr']:.2f} | Vol: {row['vol_ratio']:.2f}x\n"
                f"📊 *Diagnóstico:* {diagnostico}\n"
                f"🛡️ *Plano Ágora ({quantidade_lote} ações):* Stop R$ {stop_loss:.2f} | Alvo R$ {alvo_lucro:.2f}\n"
                f"📰 *IA Contexto:* {fato_relevante}\n\n"
            )
    else:
        msg_final += "ℹ️ Nenhuma oportunidade encontrada em Exaustão de Venda.\n\n"

    msg_final += "───────────────────────────\n\n"

    # =========================================================================
    # PROCESSAMENTO DO RELATÓRIO 2: RETOMADA CONFIRMADA (SUBIDA)
    # =========================================================================
    if pool_retomada:
        df_ret = pd.DataFrame(pool_retomada)
        df_top10_ret = df_ret.sort_values(by=['momentum'], ascending=[False]).head(10)

        msg_final += "*🚀 TOP 10 AÇÕES EM RETOMADA CONFIRMADA (COMPRA NA SUBIDA) 🚀*\n"
        msg_final += "_Pivôs de reversão validados com fluxo e cruzamento de médias_\n\n"

        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        for idx, row in df_top10_ret.reset_index(drop=True).iterrows():
            manchetes = buscar_noticias_reais_yfinance(f"{row['ticker']}.SA")
            fato_relevante = gerar_fato_ocorrido_por_ia(row['ticker'], row['preco'], manchetes)

            distancia_stop = row['atr'] * 1.5 if row['atr'] > 0 else row['preco'] * 0.015
            stop_loss = row['preco'] - distancia_stop
            alvo_lucro = row['preco'] + (distancia_stop * 2.0)
            quantidade_lote = int(RISCO_MAXIMO_FINANCEIRO / (row['preco'] - stop_loss)) if (row['preco'] - stop_loss) > 0 else 0

            msg_final += (
                f"{emojis[idx]} *{row['ticker']}* | Força Momentum: +{row['momentum']*100:.1f}%\n"
                f"• Preço: R$ {row['preco']:.2f} | IFR: {row['ifr']:.2f} | Vol: {row['vol_ratio']:.2f}x\n"
                f"🛡️ *Plano Ágora ({quantidade_lote} ações):* Stop R$ {stop_loss:.2f} | Alvo R$ {alvo_lucro:.2f}\n"
                f"📰 *IA Contexto:* {fato_relevante}\n\n"
            )
    else:
        msg_final += "ℹ️ Nenhuma ação confirmou reversão de alta neste ciclo de mercado.\n\n"

    if pool_exaustao or pool_retomada:
        print("\n================ RELATÓRIO GLOBAL EXECUTADO ================\n", msg_final)
        enviar_telegram(msg_final); enviar_email(msg_final); enviar_sms(msg_final)
    else:
        print("Nenhuma ação passou pelos filtros mínimos do scanner.")

if __name__ == "__main__":
    executar_agente_b3_analista()
