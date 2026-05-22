from telethon import TelegramClient, events

from services import detection_service, notification_service
from services.scrapers.db_context import scraper_db_session

DANGER_CHANNELS = [
    "your_channel_slug_1",
    "your_channel_slug_2",
]


async def start_watcher(client: TelegramClient, db):

    @client.on(events.NewMessage(chats=DANGER_CHANNELS))
    async def handler(event):
        try:
            if not event.message.photo and not event.message.document:
                return

            media_bytes = await event.download_media(bytes)

            if not isinstance(media_bytes, bytes):
                return

            with scraper_db_session(db) as session:
                match = detection_service.detect_suspect_image_bytes(session, media_bytes)
            if match:
                chat = await event.get_chat()
                chat_title = getattr(chat, "title", "Unknown")
                notification_service.queue_radar_alert(
                    user_id=match.user_id,
                    suspect_url=f"Telegram: {chat_title}",
                    image_url="[Telegram — no direct URL]",
                    platform="Telegram",
                    context=(
                        f"Message ID: {event.message.id}; "
                        f"{detection_service.describe_match(match)}"
                    ),
                )

        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}")

    await client.run_until_disconnected()  # type: ignore[func-returns-value]
