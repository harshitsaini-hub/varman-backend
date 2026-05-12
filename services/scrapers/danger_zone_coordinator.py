import asyncio
import threading
from services.scrapers import reddit_scraper, telegram_scraper, fourchan_scraper

def launch_all_danger_zone_watchers(db, reddit_client, telegram_client):
    """
    Each scraper runs in its own thread/event loop.
    They are independent — one crashing doesn't kill the others.
    """
    # Reddit: blocking stream, runs in its own thread
    reddit_thread = threading.Thread(
        target=reddit_scraper.start_watcher,
        args=(reddit_client, db),
        daemon=True,
        name="reddit-watcher"
    )
    
    # 4chan: polling loop, runs in its own thread
    fourchan_thread = threading.Thread(
        target=fourchan_scraper.start_watcher,
        args=(db,),
        daemon=True,
        name="fourchan-watcher"
    )
    
    # Telegram: async, runs in its own event loop thread
    telegram_thread = threading.Thread(
        target=_run_telegram_async,
        args=(telegram_client, db),
        daemon=True,
        name="telegram-watcher"
    )
    
    for t in [reddit_thread, fourchan_thread, telegram_thread]:
        t.start()
        print(f"[DANGER ZONE] Started {t.name}")

def _run_telegram_async(client, db):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(telegram_scraper.start_watcher(client, db))