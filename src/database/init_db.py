from sqlalchemy import text
from db_models import get_engine, create_tables

if __name__ == "__main__":
    engine = get_engine(host="localhost")

    # --- 1️⃣ Alte Tabellen löschen ---
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS selection_moves CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS selection_games CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS selfplay_moves CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS selfplay_games CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS training_moves CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS training_games CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS models CASCADE;"))
        conn.commit()
    print("✅ Old tables dropped.")

    # --- 2️⃣ Neue Tabellen erstellen ---
    create_tables(engine)
    print("✅ DB tables created.")
