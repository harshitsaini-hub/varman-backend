from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def scraper_db_session(db_source: Any | Callable[[], Any]) -> Iterator[Any]:
    """Give each scraper event an isolated DB session when a factory is provided."""
    if callable(db_source):
        db = db_source()
        should_close = True
    else:
        db = db_source
        should_close = False

    try:
        yield db
    finally:
        if should_close:
            db.close()
