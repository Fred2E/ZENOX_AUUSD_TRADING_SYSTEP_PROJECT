# zeno_gap_classifier.py

import os
import pandas as pd
from datetime import datetime, timedelta
import logging

DATA_ROOT = os.path.join("data", "live", "XAUUSD")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
CSV_TEMPLATE = "XAUUSD_{}_LIVE.csv"
WEEKEND_START = 21, 59  # Friday 21:59 UTC
WEEKEND_END = 22, 0     # Sunday 22:00 UTC

def is_weekend_gap(start, end):
    # Returns True if gap covers Friday 22:00 to Sunday 22:00 (UTC)
    # Handles gaps that cross weekends
    curr = start
    while curr < end:
        # Check if in weekend window
        if curr.weekday() == 4 and (curr.hour, curr.minute) >= WEEKEND_START:
            next_sunday = curr + timedelta(days=(6-curr.weekday())%7)
            sunday_22 = datetime(next_sunday.year, next_sunday.month, next_sunday.day, 22, 0, tzinfo=curr.tzinfo)
            if end > sunday_22:
                return True
        curr += timedelta(minutes=1)
    return False

def classify_gap(start, end):
    # Simple classifier for now: weekend or unexpected
    if is_weekend_gap(start, end):
        return "WEEKEND"
    return "UNEXPECTED"

def check_gaps_with_type(tf):
    path = os.path.join(DATA_ROOT, tf, CSV_TEMPLATE.format(tf))
    if not os.path.exists(path):
        print(f"[{tf}] Missing file: {path}")
        return
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime')
    freq = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}[tf]
    for i in range(1, len(df)):
        diff = (df.iloc[i]['Datetime'] - df.iloc[i-1]['Datetime']).total_seconds() / 60
        if diff > freq * 1.5:
            start, end = df.iloc[i-1]['Datetime'], df.iloc[i]['Datetime']
            gap_type = classify_gap(start, end)
            print(f"[{tf}] GAP: {start} -> {end} = {diff}min [{gap_type}]")

def main():
    for tf in TIMEFRAMES:
        check_gaps_with_type(tf)

if __name__ == "__main__":
    main()
