from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all Varman ORM models.

    Every model file (``user.py``, ``protected_image.py``, …) imports
    ``Base`` from here and subclasses it.  ``database.init_db()`` then
    calls ``Base.metadata.create_all`` to bootstrap the schema.
    """
    pass


# Re-export models so that ``from app.models import User, ProtectedImage``
# works and, more importantly, so that ``Base.metadata`` knows about every
# table when ``create_all`` is called.

from app.models.user import User  # noqa: E402, F401
from app.models.protected_image import ProtectedImage  # noqa: E402, F401

__all__ = ["Base", "User", "ProtectedImage"]
