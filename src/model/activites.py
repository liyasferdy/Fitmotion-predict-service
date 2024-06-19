from datetime import datetime
from sqlalchemy import DateTime, Date, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP, DATE, REAL, SMALLINT

class BaseModel(DeclarativeBase):
    pass

class Records(BaseModel):
    __tablename__ = 'records'
    
    id: Mapped[int] = mapped_column(SMALLINT, primary_key=True, unique=True, server_default=text("nextval('records_id_seq'::regclass)"))
    fk_user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), nullable=False)
    walk_time_min: Mapped[float] = mapped_column(REAL)
    jogging_time_min: Mapped[float] = mapped_column(REAL)
    stand_time_min: Mapped[float] = mapped_column(REAL)
    sit_time_min: Mapped[float] = mapped_column(REAL)
    upstair_time_min: Mapped[float] = mapped_column(REAL)
    downstair_time_min: Mapped[float] = mapped_column(REAL)
    created_at: Mapped[Date] = mapped_column(DATE, nullable=False, default=datetime.now())
    updated_at: Mapped[DateTime] = mapped_column(TIMESTAMP)