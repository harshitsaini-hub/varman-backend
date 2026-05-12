import asyncio
import threading

from telethon import TelegramClient

from services.scrapers import fourchan_scraper, reddit_scraper, telegram_scraper


def launch_all_danger_zone_watchers(
    db,
    reddit_client,
    tg_session_name: str,
    tg_api_id: int,
    tg_api_hash: str,
):

    reddit_thread = threading.Thread(
        target=reddit_scraper.start_watcher,
        args=(reddit_client, db),
        daemon=True,
        name="reddit-watcher",
    )

    fourchan_thread = threading.Thread(
        target=fourchan_scraper.start_watcher,
        args=(db,),
        daemon=True,
        name="fourchan-watcher",
    )

    telegram_thread = threading.Thread(
        target=_run_telegram_async,
        args=(tg_session_name, tg_api_id, tg_api_hash, db),
        daemon=True,
        name="telegram-watcher",
    )

    for t in [reddit_thread, fourchan_thread, telegram_thread]:
        t.start()
        print(f"[DANGER ZONE] Started {t.name}")


def _run_telegram_async(session_name, api_id, api_hash, db):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = TelegramClient(session_name, api_id, api_hash)

    try:
        with client:
            print("[TELEGRAM] Client connected. Starting watcher...")
            loop.run_until_complete(telegram_scraper.start_watcher(client, db))
    except Exception as e:
        print(f"[TELEGRAM ERROR] Thread crashed: {e}")
