import time
import basc_py4chan
from utils.hashing import compute_phash_from_url
from services import db_service, notification_service

DANGER_BOARDS = ["b", "gif"]  # Extend based on threat intelligence
POLL_INTERVAL_SECONDS = 300   # 4chan threads move fast. 5 min is reasonable.

def start_watcher(db):
    print(f"[4CHAN] Polling boards: {', '.join(DANGER_BOARDS)} every {POLL_INTERVAL_SECONDS}s")
    seen_post_ids = set()  # In-memory dedup. Prevents reprocessing same post each poll cycle.
    
    while True:
        try:
            for board_name in DANGER_BOARDS:
                board = basc_py4chan.Board(board_name)
                all_threads = board.get_all_threads(expand=True)
                
                for thread in all_threads:
                    for post in thread.posts:
                        if post.post_id in seen_post_ids:
                            continue
                        if not post.has_file or not post.file.is_image:
                            continue
                        
                        seen_post_ids.add(post.post_id)
                        
                        phash = compute_phash_from_url(post.file.file_url)
                        if phash is None:
                            continue
                        
                        match = db_service.lookup_phash_global(db, phash, threshold=10)
                        if match:
                            notification_service.send_radar_alert(
                                user_id=match.user_id,
                                suspect_url=thread.url,
                                image_url=post.file.file_url,
                                platform="4chan",
                                context=f"/{board_name}/ — Thread {thread.id}"
                            )
                            print(f"[4CHAN HIT] /{board_name}/ Thread {thread.id}. User {match.user_id} notified.")
            
            # Prevent seen_post_ids growing forever. 4chan posts expire anyway.
            if len(seen_post_ids) > 50000:
                seen_post_ids = set(list(seen_post_ids)[-25000:])
                
        except Exception as e:
            print(f"[4CHAN ERROR] {e} — retrying in {POLL_INTERVAL_SECONDS}s")
        
        time.sleep(POLL_INTERVAL_SECONDS)