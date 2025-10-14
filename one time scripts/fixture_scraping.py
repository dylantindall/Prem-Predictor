# Scrape the current seasons available data from the ESPN website
# into the fixtures_2025_2026.csv file.

import requests
from bs4 import BeautifulSoup
import csv
import re
from typing import Optional

def fetch_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.text

def parse_espn_fixtures(html):
    soup = BeautifulSoup(html, "html.parser")
    fixtures = []

    # Try a few likely content containers that ESPN uses; fallback to whole document text
    candidate_selectors = [
        ("div", {"class": "article-body"}),
        ("div", {"class": "article-body__content"}),
        ("article", {}),
        ("section", {"class": "article"}),
    ]

    content = None
    for tag, attrs in candidate_selectors:
        content = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
        if content:
            break
    if not content:
        content = soup

    # Extract normalized lines of text (collapse internal whitespace, keep logical lines)
    raw_text = content.get_text("\n", strip=True)
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.split("\n")]
    lines = [line for line in lines if line]

    weekday_regex = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    month_regex = r"Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December"
    # Date lines usually include weekday + month + day + year. Be lenient on punctuation.
    date_pattern = re.compile(rf"(?:{weekday_regex}).*(?:{month_regex}).*\b20\d{{2}}\b", re.I)
    # Matchweek headings like "Matchweek 1"
    matchweek_pattern = re.compile(r"match\s*week\s*(\d+)|matchweek\s*(\d+)", re.I)

    # Helper: normalize date like "Saturday, Aug. 23, 2025" -> "2025/08/23"
    month_to_num = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    date_extract_pattern = re.compile(rf"(?:{weekday_regex})\s*,?\s*([A-Za-z\.]+)\s*\.?\s*(\d{{1,2}})\s*,?\s*(20\d{{2}})", re.I)

    def normalize_date(date_text: Optional[str]) -> Optional[str]:
        if not date_text:
            return None
        m = date_extract_pattern.search(date_text)
        if not m:
            return None
        month_name, day_str, year_str = m.group(1), m.group(2), m.group(3)
        month_key = month_name.replace('.', '').lower()
        month_num = month_to_num.get(month_key)
        if not month_num:
            return None
        day = int(day_str)
        year = int(year_str)
        return f"{year:04d}/{month_num:02d}/{day:02d}"

    current_date = None
    current_matchweek: Optional[int] = None

    for line in lines:
        # Detect and set current date
        if date_pattern.search(line):
            current_date = normalize_date(line)
            continue

        # Detect and set current matchweek
        mw = matchweek_pattern.search(line)
        if mw:
            # pick whichever group matched
            num_str = mw.group(1) or mw.group(2)
            try:
                current_matchweek = int(num_str)
            except (TypeError, ValueError):
                current_matchweek = None
            continue

        # Normalize common separators in match lines and detect fixtures
        normalized = line
        normalized = normalized.replace(" vs. ", " vs ")
        normalized = normalized.replace(" v ", " vs ")

        if " vs " in normalized:
            home, away_part = normalized.split(" vs ", 1)

            # Remove trailing time, TV, venue, or notes from away part
            away = re.split(r"\s*(?:\(|\[|,|-\s*KO|KO\b)", away_part)[0].strip()

            # Clean trailing markers like '*' and stray punctuation
            home = re.sub(r"\*+\s*$", "", home.strip())
            away = re.sub(r"\*+\s*$", "", away.strip())

            if home and away:
                fixtures.append({
                    "match_week": current_matchweek,
                    "date": current_date,
                    "home_team": home,
                    "away_team": away
                })

    # Fallback: if matchweek headings were not detected, infer sequentially (10 fixtures per matchweek)
    if fixtures and all(fx.get("match_week") in (None, "", 0) for fx in fixtures):
        week = 1
        count_in_week = 0
        for fx in fixtures:
            fx["match_week"] = week
            count_in_week += 1
            if count_in_week == 10:
                week += 1
                count_in_week = 0

    return fixtures

def save_to_csv(fixtures, filename):
    keys = ["match_week", "date", "home_team", "away_team"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for fx in fixtures:
            writer.writerow(fx)

def main():
    url = "https://www.espn.co.uk/football/story/_/id/45522470/premier-league-fixtures-schedule-2025-26-full"
    html = fetch_html(url)
    fixtures = parse_espn_fixtures(html)
    save_to_csv(fixtures, "espn_premier_league_2025_26.csv")
    print(f"Found {len(fixtures)} fixtures.")

if __name__ == "__main__":
    main()