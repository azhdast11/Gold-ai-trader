import os
import re
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from urllib.parse import quote
from datetime import datetime


# ==========================================
# تنظیمات تلگرام
# ==========================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# ==========================================
# ارسال گزارش به تلگرام
# ==========================================

def send_telegram(message):

    if not BOT_TOKEN:
        raise ValueError(
            "خطا: TELEGRAM_BOT_TOKEN در GitHub Secrets خالی است."
        )

    if not CHAT_ID:
        raise ValueError(
            "خطا: TELEGRAM_CHAT_ID در GitHub Secrets خالی است."
        )

    # بررسی ساختار توکن بدون نمایش خود آن
    if not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", BOT_TOKEN):
        raise ValueError(
            "ساختار TELEGRAM_BOT_TOKEN اشتباه است. "
            "فقط توکن واقعی BotFather را در GitHub ثبت کن."
        )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    print("Checking Telegram connection...")

    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=30
        )

        try:
            result = response.json()
        except ValueError:
            raise RuntimeError(
                f"پاسخ تلگرام JSON معتبر نیست. HTTP: {response.status_code}"
            )

        if response.status_code != 200 or not result.get("ok"):
            description = result.get(
                "description",
                "خطای نامشخص تلگرام"
            )

            error_code = result.get(
                "error_code",
                response.status_code
            )

            raise RuntimeError(
                f"خطای تلگرام: {error_code} - {description}"
            )

        print("Telegram report sent successfully!")
        print("Message ID:", result["result"]["message_id"])

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "ارتباط با تلگرام بیش از حد طول کشید."
        )

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "اتصال به سرور تلگرام برقرار نشد."
        )


# ==========================================
# دریافت اطلاعات قیمت طلا
# ==========================================

def get_gold_data():

    # نماد GC=F ابتدا بررسی می‌شود
    symbols = ["GC=F", "XAUUSD=X"]

    for symbol in symbols:

        print(f"Downloading gold data: {symbol}")

        try:
            data = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if data is None or data.empty:
                print(f"No data for {symbol}")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            required_columns = ["Close", "High", "Low"]

            if not all(
                column in data.columns
                for column in required_columns
            ):
                print(f"Missing columns for {symbol}")
                continue

            data = data.dropna(subset=required_columns)

            if len(data) >= 60:
                print(f"Gold data received: {symbol}")
                return data, symbol

        except Exception:
            print(f"Download failed for {symbol}")

    raise RuntimeError(
        "دریافت اطلاعات قیمت طلا از Yahoo Finance ناموفق بود."
    )


# ==========================================
# محاسبه اندیکاتورها و تحلیل
# ==========================================

def calculate_analysis(data):

    close = data["Close"].squeeze()
    high = data["High"].squeeze()
    low = data["Low"].squeeze()

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal_line = macd.ewm(
        span=9,
        adjust=False
    ).mean()

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

    values = [
        price,
        current_ema20,
        current_ema50,
        current_rsi,
        current_macd,
        current_signal,
        current_atr
    ]

    if not all(np.isfinite(value) for value in values):
        raise RuntimeError(
            "برخی اندیکاتورها قابل محاسبه نیستند."
        )

    if current_atr <= 0:
        raise RuntimeError("مقدار ATR معتبر نیست.")

    direction = "بدون معامله"

    entry = None
    stop_loss = None
    take_profit = None

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


# ==========================================
# ترجمه اخبار به فارسی
# ==========================================

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

        return "".join(
            part[0]
            for part in result[0]
            if part[0]
        )

    except Exception:
        return text


# ==========================================
# دریافت اخبار طلا
# ==========================================

def get_gold_news():

    url = (
        "https://news.google.com/rss/search?q="
        + quote(
            "gold OR XAUUSD OR Federal Reserve OR inflation"
        )
        + "&hl=en-US&gl=US&ceid=US:en"
    )

    try:

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)

        items = root.findall(".//item")

        news = []

        for item in items[:5]:

            title = item.findtext(
                "title",
                ""
            ).strip()

            if title:
                news.append(
                    translate_to_persian(title)
                )

        if not news:
            return ["خبر جدیدی دریافت نشد."]

        return news

    except Exception as error:

        print("News retrieval failed:", type(error).__name__)

        return [
            "دریافت اخبار در حال حاضر امکان‌پذیر نیست."
        ]


# ==========================================
# ساخت گزارش روزانه
# ==========================================

def create_report(analysis, symbol, news):

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

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

        report.append(
            "شرایط ورود فراهم نیست؛ فعلاً معامله نکن."
        )

    report.extend([
        "",
        "📰 اخبار مهم مرتبط با طلا:"
    ])

    for index, title in enumerate(news, 1):
        report.append(
            f"{index}. {title}"
        )

    report.extend([
        "",
        "⚠️ این گزارش صرفاً تحلیلی و آموزشی است.",
        "سیگنال قطعی نیست؛ قبل از معامله مدیریت ریسک را رعایت کن."
    ])

    return "\n".join(report)


# ==========================================
# اجرای اصلی برنامه
# ==========================================

def main():

    print("================================")
    print("GOLD AI TRADER STARTED")
    print("================================")

    print("Downloading gold prices...")

    data, symbol = get_gold_data()

    print("Calculating indicators...")

    analysis = calculate_analysis(data)

    print("Getting news...")

    news = get_gold_news()

    print("Creating report...")

    report = create_report(
        analysis,
        symbol,
        news
    )

    print("Report created successfully.")

    print("Sending report to Telegram...")

    send_telegram(report)

    print("================================")
    print("GOLD AI TRADER FINISHED")
    print("TELEGRAM DELIVERY SUCCESSFUL")
    print("================================")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("================================")
        print("PROGRAM FAILED")
        print("Error type:", type(error).__name__)
        print("Error message:", str(error))
        print("================================")

        raise
