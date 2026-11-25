"""Part B step 3: collect real-world used-car listings from moteur.ma (used cars, "achat-voiture-occasion").

Output: data/real_world/raw/moteur_<YYYY-MM-DD>.csv, with the SAME columns and value formats as data/raw/cars_dataframe.csv
(Brand, Model, Year, Condition, Mileage, Gearbox, Fiscal Power, Fuel, Equipment, Number of Doors, Price), so the saved
models can predict them directly (src/shared/saved_models.py: predict_prices). Extra columns: listing_id, url, posted,
scraped_at, Mileage_km.

Run it with the Run button, or from the GitHub Action .github/workflows/collect_listings.yml (once a week).

Being polite to the website:
  - robots.txt allows these pages (checked 2026-11-24); one request every DELAY_SECONDS seconds, at most MAX_PAGES list pages
    (30 listings each) per run, and listings already saved in an earlier run are skipped.
  - Only the car's characteristics and price are kept: no seller name, phone number, description text or photos.
  - moteur.ma's terms say its content and databases may not be reproduced without permission, so do not publish the
    collected CSVs (keep the repository private, or ask moteur.ma first). The GitHub Action refuses to run on a public repository.

Not on the page: Condition (left empty; the prediction uses the most common training condition). Mileage is given in km
and is converted to the dataset's ranges ("160 000 - 169 999").
"""
import csv
import html
import os
import re
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("REAL_WORLD_OUT_DIR", ROOT_DIR / "data" / "real_world" / "raw"))

LIST_URL = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion?page={page}"
MAX_PAGES = int(os.environ.get("MAX_PAGES", 5))           # list pages per run (30 listings each)
MAX_LISTINGS = int(os.environ.get("MAX_LISTINGS", 1000))  # safety cap per run
DELAY_SECONDS = 5
USER_AGENT = "Mozilla/5.0 (compatible; student research project; used-car price study)"

COLUMNS = ["listing_id", "url", "posted", "scraped_at", "Brand", "Model", "Year", "Condition", "Mileage", "Mileage_km",
           "Gearbox", "Fiscal Power", "Fuel", "Equipment", "Number of Doors", "Price"]

# moteur.ma (French) -> the values used in the training data
FUEL = {"diesel": "Diesel", "essence": "Petrol", "hybride": "Hybrid", "hybride rechargeable": "Hybrid",
        "electrique": "Electrique", "électrique": "Electrique", "gpl": "LPG"}
GEARBOX = {"automatique": "Automatic", "manuelle": "Manual"}
NOT_EQUIPMENT = {"état du véhicule"}  # shown in the options list on every page, not a piece of equipment


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "fr"})
    with urllib.request.urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8", "ignore")
    time.sleep(DELAY_SECONDS)
    return page


def text(fragment):
    """HTML fragment -> plain text with single spaces."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def listing_urls(list_page):
    """Detail-page URLs on one list page, in page order, without duplicates."""
    urls = re.findall(r'href="(https://www\.moteur\.ma/fr/voiture/achat-voiture-occasion/detail-annonce/\d+/[^"]+)"', list_page)
    return list(dict.fromkeys(urls))


def mileage_range(km):
    """160000 -> '160 000 - 169 999', in the training data's ranges (5k wide below 100k, 10k up to 200k, 50k up to 500k)."""
    if km >= 500_000:
        return "Plus de 500 000"
    width = 5_000 if km < 100_000 else 10_000 if km < 200_000 else 50_000
    low = km // width * width
    fmt = lambda n: f"{n:,}".replace(",", " ")
    return f"{fmt(low)} - {fmt(low + width - 1)}"


def parse_listing(page, url):
    """One detail page -> one row (dict), or a string saying which needed value is missing (e.g. no price: 'Appeler pour le prix')."""
    fields = {}
    for label, value in re.findall(r'<span class="font-weight-bold">\s*([^<]+?)\s*</span>\s*</td>\s*<td>(.*?)</td>', page, flags=re.DOTALL):
        fields[label.rstrip(":").strip().lower()] = text(value)

    price = re.search(r'ad-hero-price-col">\s*([\d][\d,.\s]*)\s*MAD', page)
    km = re.sub(r"\D", "", fields.get("kilométrage", ""))
    year = re.search(r"\d{4}", fields.get("année", ""))
    fuel = FUEL.get(fields.get("motorisation", "").lower())
    gearbox = GEARBOX.get(fields.get("transmission", "").lower())
    needed = {"price": price, "mileage": km, "year": year, "brand": fields.get("marque"), "model": fields.get("modèle"),
              f"fuel ({fields.get('motorisation', '')!r})": fuel, f"gearbox ({fields.get('transmission', '')!r})": gearbox}
    missing = [name for name, value in needed.items() if not value]
    if missing:
        return "missing " + ", ".join(missing)

    options = page.split('<h4 class="mb-4">options</h4>', 1)
    equipment = []
    if len(options) == 2:
        section = options[1].split("</h4>", 1)[0]  # stop at the next section title
        equipment = [text(item) for item in re.findall(r'fa-check[^>]*></i>(.*?)</div>', section, flags=re.DOTALL)]
        equipment = [item for item in equipment if item and item.lower() not in NOT_EQUIPMENT]

    fiscal = re.sub(r"\D", "", fields.get("puissance fiscale", ""))
    doors = re.sub(r"\D", "", fields.get("nombre de portes", ""))
    posted = re.search(r'fa-calendar"></i>\s*([^<]+?)\s*</span>', page)
    km = int(km)
    return {
        "listing_id": re.search(r"/detail-annonce/(\d+)/", url).group(1),
        "url": url,
        "posted": posted.group(1) if posted else "",
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "Brand": fields["marque"],
        "Model": fields["modèle"],
        "Year": year.group(0),
        "Condition": "",  # not shown on moteur.ma
        "Mileage": mileage_range(km),
        "Mileage_km": km,
        "Gearbox": gearbox,
        "Fiscal Power": "" if not fiscal else ("Plus de 41 CV" if int(fiscal) > 41 else f"{int(fiscal)} CV"),
        "Fuel": fuel,
        "Equipment": repr(equipment),  # same format as the training data: "['ABS', 'Climatisation']"
        "Number of Doors": doors,
        "Price": int(re.sub(r"\D", "", price.group(1))),
    }


def already_saved():
    """listing_ids saved by earlier runs (any CSV in OUT_DIR)."""
    ids = set()
    for path in OUT_DIR.glob("moteur_*.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            ids.update(row["listing_id"] for row in csv.DictReader(f))
    return ids


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = already_saved()
    out_path = OUT_DIR / f"moteur_{date.today().isoformat()}.csv"
    new_file = not out_path.exists()
    saved = skipped = 0
    print(f"{len(seen)} listings saved before; up to {MAX_PAGES} list pages, {DELAY_SECONDS} s between requests")

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for page_number in range(1, MAX_PAGES + 1):
            urls = listing_urls(fetch(LIST_URL.format(page=page_number)))
            if not urls:
                break
            for url in urls:
                listing_id = re.search(r"/detail-annonce/(\d+)/", url).group(1)
                if listing_id in seen:
                    continue
                if saved >= MAX_LISTINGS:
                    break
                seen.add(listing_id)
                try:
                    row = parse_listing(fetch(url), url)
                except Exception as error:  # one broken page should not stop the run
                    row = f"error: {error}"
                if isinstance(row, str):
                    print(f"  skipped {url}: {row}")
                    skipped += 1
                    continue
                writer.writerow(row)
                f.flush()
                saved += 1
            print(f"page {page_number}: {saved} saved so far, {skipped} skipped")
            if saved >= MAX_LISTINGS:
                break

    print(f"Saved {saved} new listings in {out_path}")


if __name__ == "__main__":
    main()
