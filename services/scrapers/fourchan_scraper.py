import time

import basc_py4chan

from services import detection_service, notification_service
from services.scrapers.db_context import scraper_db_session

DANGER_BOARDS = ["b", "gif"]
POLL_INTERVAL_SECONDS = 300


def start_watcher(db):
    seen_post_ids: set = set()

    while True:
        try:
            for board_name in DANGER_BOARDS:
                board = basc_py4chan.Board(board_name)
                all_threads = board.get_all_threads(expand=True)

                for thread in all_threads:
                    for post in thread.posts: 
                        post_id = getattr(post, "post_id", None)
                        if post_id in seen_post_ids:
                            continue

                        has_file = getattr(post, "has_file", False)
                        if not has_file:
                            continue

                        post_file = getattr(post, "file", None)
                        if post_file is None:
                            continue

                        is_image = getattr(post_file, "is_image", False)
                        if not is_image:
                            continue

                        file_url = getattr(post_file, "file_url", None)
                        if not file_url:
                            continue

                        seen_post_ids.add(post_id)

                        with scraper_db_session(db) as session:
                            match = detection_service.detect_suspect_image_url(session, file_url)
                        if match:
                            notification_service.queue_radar_alert(
                                user_id=match.user_id,
                                suspect_url=thread.url,
                                image_url=file_url,
                                platform="4chan",
                                context=(
                                    f"/{board_name}/ — Thread {getattr(thread, 'id', '?')}; "
                                    f"{detection_service.describe_match(match)}"
                                ),
                            )

            if len(seen_post_ids) > 50000:
                seen_post_ids = set(list(seen_post_ids)[-25000:])

        except Exception as e:
            print(f"[4CHAN ERROR] {e}")

        time.sleep(POLL_INTERVAL_SECONDS)
