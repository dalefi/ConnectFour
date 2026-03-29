from sqlalchemy import (
    Column, Integer, Float, DateTime, ForeignKey, create_engine, String, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()

class TrainingGame(Base):
    __tablename__ = "training_games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    winner = Column(Integer, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    move_count = Column(Integer, nullable=False, default=0)

    model_tag = Column(String, nullable=False)

    moves = relationship(
        "TrainingMove",
        back_populates="game",
        cascade="all, delete-orphan"
    )



class TrainingMove(Base):
    __tablename__ = "training_moves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("training_games.id"), nullable=False)

    move_number = Column(Integer, nullable=False)
    board_state = Column(JSON, nullable=False)
    policy = Column(JSON, nullable=False)
    nn_policy = Column(JSON, nullable=False)
    value = Column(Float, nullable=False)
    nn_evaluation = Column(Float, nullable=True)
    current_player = Column(Integer, nullable=False)

    game = relationship("TrainingGame", back_populates="moves")


class SelfplayGame(Base):
    __tablename__ = "selfplay_games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    winner = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    move_count = Column(Integer, nullable=False, default=0)

    model_1_tag = Column(String, nullable=False)
    model_2_tag = Column(String, nullable=False)

    moves = relationship(
        "SelfplayMove",
        back_populates="game",
        cascade="all, delete-orphan"
    )



class SelfplayMove(Base):
    __tablename__ = "selfplay_moves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("selfplay_games.id"), nullable=False)

    move_number = Column(Integer, nullable=False)
    board_state = Column(JSON, nullable=False)
    policy = Column(JSON, nullable=False)
    nn_policy = Column(JSON, nullable=False)
    value = Column(Float, nullable=False)
    nn_evaluation = Column(Float, nullable=True)
    current_player = Column(Integer, nullable=False)
    model_role = Column(String, nullable=False)  # "current" | "updated"

    game = relationship("SelfplayGame", back_populates="moves")


# --- DB Helper ---
def get_engine(user="daniel", password="connectfour", host="localhost", port=5432, db="connectfour"):
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, echo=False, future=True)


def create_tables(engine):
    Base.metadata.create_all(engine)


def get_session(engine):
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return session()
