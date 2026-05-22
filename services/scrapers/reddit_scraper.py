import praw

from services import detection_service, notification_service
from services.scrapers.db_context import scraper_db_session

DANGER_SUBREDDITS = [
    "SFWdeepfakes",
    "MediaSynthesis",
    "wormwood_studios",
    "unstableaiart",
]


def start_watcher(reddit: praw.Reddit, db):
    print(f"[REDDIT] Watching: {', '.join(DANGER_SUBREDDITS)}")
    target = reddit.subreddit("+".join(DANGER_SUBREDDITS))

    for submission in target.stream.submissions(skip_existing=True):
        try:
            if not submission.url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                continue

            with scraper_db_session(db) as session:
                match = detection_service.detect_suspect_image_url(session, submission.url)
            if match:
                notification_service.queue_radar_alert(
                    user_id=match.user_id,
                    suspect_url=f"https://reddit.com{submission.permalink}",
                    image_url=submission.url,
                    platform="Reddit",
                    context=(
                        f"Posted in r/{submission.subreddit.display_name}; "
                        f"{detection_service.describe_match(match)}"
                    ),
                )
                print(f"[REDDIT HIT] Match found. User {match.user_id} notified.")
        except Exception as e:
            print(f"[REDDIT ERROR] {e} — continuing stream")
            continue
