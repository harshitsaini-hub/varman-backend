from telethon import TelegramClient, events
from utils.hashing import compute_phash_from_bytes
from services import db_service, notification_service

DANGER_CHANNELS = [
    # Add actual channel slugs/IDs from your threat intelligence
    # Example placeholders:
    "deepfake_channel_1",
    "deepfake_channel_2",
]

async def start_watcher(client: TelegramClient, db):
    print(f"[TELEGRAM] Watching {len(DANGER_CHANNELS)} channels")
    
    @client.on(events.NewMessage(chats=DANGER_CHANNELS))
    async def handler(event):
        try:
            if not event.message.photo and not event.message.document:
                return
            
            media_bytes = await event.download_media(bytes)
            if media_bytes is None:
                return
            
            phash = compute_phash_from_bytes(media_bytes)
            match = db_service.lookup_phash_global(db, phash, threshold=10)
            
            if match:
                chat = await event.get_chat()
                notification_service.send_radar_alert(
                    user_id=match.user_id,
                    suspect_url=f"Telegram channel: {getattr(chat, 'title', 'Unknown')}",
                    image_url="[Telegram — no direct URL]",
                    platform="Telegram",
                    context=f"Message ID: {event.message.id}"
                )
                print(f"[TELEGRAM HIT] Match found. User {match.user_id} notified.")
        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}")
    
    await client.run_until_disconnected()