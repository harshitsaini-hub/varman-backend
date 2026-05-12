import praw
from utils.hashing import compute_phash_from_url
from services import db_service, notification_service

DANGER_SUBREDDITS = [
    "SFWdeepfakes",
    "MediaSynthesis", 
    "wormwood_studios",
    "unstableaiart",       # Add/remove based on your threat intelligence
]

def start_watcher(reddit: praw.Reddit, db):
    print(f"[REDDIT] Watching: {', '.join(DANGER_SUBREDDITS)}")
    target = reddit.subreddit("+".join(DANGER_SUBREDDITS))
    
    for submission in target.stream.submissions(skip_existing=True):
        try:
            if not submission.url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                continue
            
            phash = compute_phash_from_url(submission.url)
            if phash is None:
                continue
                
            match = db_service.lookup_phash_global(db, phash, threshold=10)
            if match:
                notification_service.send_radar_alert(
                    user_id=match.user_id,
                    suspect_url=f"https://reddit.com{submission.permalink}",
                    image_url=submission.url,
                    platform="Reddit",
                    context=f"Posted in r/{submission.subreddit.display_name}"
                )
                print(f"[REDDIT HIT] Match found. User {match.user_id} notified.")
        except Exception as e:
            print(f"[REDDIT ERROR] {e} — continuing stream")
            continue