
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Telegram settings are missing")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=30
    )
    response.raise_for_status()


def get_gold_data():
    for symbol in ["XAUUSD=X", "GC=F"]:
        try:
            data = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if data is not None and len(data) >= 60:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data, symbol
        except Exception as error:
            print(f"Error with {symbol}: {error}")

    raise RuntimeError("Could not download gold prices")


def calculate_analysis(data):
    close = data["Close"].squeeze()
    high = data["High"].squeeze()
    low = data["Low"].squeeze()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()

    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = true_range.rolling(14).mean()

    price = float(close.iloc[-1])
    current_ema20 = float(ema20.iloc[-1])
    current_ema50 = float(ema50.iloc[-1])
    current_rsi = float(rsi.iloc[-1])
    current_macd = float(macd.iloc[-1])
    current_signal = float(signal_line.iloc[-1])
    current_atr = float(atr.iloc[-1])

    if not np.isfinite(current_atr) or current_atr <= 0:
        raise RuntimeError("Technical indicators are incomplete")

    direction = "بدون معامله"
    entry = stop_loss = take_profit = None

    buy_condition = (
        price > current_ema20 > current_ema50
        and 50 < current_rsi < 68
        and current_macd > current_signal
    )

    sell_condition = (
        price < current_ema20 < current_ema50
        and 32 < current_rsi < 50
        and current_macd < current_signal
    )

    if buy_condition:
        direction = "خرید (BUY)"
        entry = price
        stop_loss = price - (1.5 * current_atr)
        take_profit = price + (3 * current_atr)

    elif sell_condition:
        direction = "فروش (SELL)"
        entry = price
        stop_loss = price + (1.5 * current_atr)
        take_profit = price - (3 * current_atr)

    return {
        "price": price,
        "rsi": current_rsi,
        "ema20": current_ema20,
        "ema50": current_ema50,
        "macd": current_macd,
        "direction": direction,
        "entry": entry,
        "stop": stop_loss,
        "target": take_profit
    }


def translate_to_persian(text):
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        response = requests.get(
            url,
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "fa",
                "dt": "t",
                "q": text
            },
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        return "".join(part[0] for part in result[0] if part[0])
    except Exception:
        return text


def get_gold_news():
    url = (
        "https://news.google.com/rss/search?q="
        + quote("gold OR XAUUSD OR Federal Reserve OR inflation")
        + "&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall(".//item")

        news = []
        for item in items[:5]:
            title = item.findtext("title", "").strip()
            if title:
                news.append(translate_to_persian(title))

        return news

    except Exception as error:
        print("News error:", error)
        return ["دریافت اخبار در حال حاضر امکان‌پذیر نیست."]


def create_report(analysis, symbol, news):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = [
        "📊 گزارش روزانه تحلیل طلا",
        f"📅 زمان گزارش: {now}",
        f"📈 نماد داده: {symbol}",
        "",
        f"💰 قیمت آخرین داده: {analysis['price']:.2f}",
        f"📉 RSI: {analysis['rsi']:.2f}",
        f"EMA 20: {analysis['ema20']:.2f}",
        f"EMA 50: {analysis['ema50']:.2f}",
        "",
        "🎯 نتیجه تحلیل:",
        analysis["direction"]
    ]

    if analysis["entry"] is not None:
        report.extend([
            "",
            f"ورود پیشنهادی: {analysis['entry']:.2f}",
            f"حد ضرر (SL): {analysis['stop']:.2f}",
            f"حد سود (TP): {analysis['target']:.2f}",
            "نسبت ریسک به بازده: 1 به 2"
        ])
    else:
        report.append("شرایط ورود مطمئن فراهم نیست؛ فعلاً معامله نکن.")

    report.extend([
        "",
        "📰 اخبار مهم مرتبط با طلا:"
    ])

    for index, title in enumerate(news, 1):
        report.append(f"{index}. {title}")

    report.extend([
        "",
        "⚠️ این گزارش صرفاً تحلیلی و آموزشی است.",
        "سیگنال قطعی نیست؛ قبل از معامله مدیریت ریسک را رعایت کن."
    ])

    return "\n".join(report)


def main():
    print("Downloading gold prices...")
    data, symbol = get_gold_data()

    print("Calculating indicators...")
    analysis = calculate_analysis(data)

    print("Getting news...")
    news = get_gold_news()

    report = create_report(analysis, symbol, news)

    print(report)
    send_telegram(report)

    print("Telegram report sent successfully.")


if __name__ == "__main__":
    main()
