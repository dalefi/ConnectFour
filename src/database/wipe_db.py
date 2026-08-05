from sqlalchemy.orm import sessionmaker

from src.database.db_models import get_engine, Base, create_tables
from src.utils import enable_utf8_console

def wipe_database(confirm=True):
    """
    Löscht alle Tabellen und erstellt sie neu.
    Setze confirm=False, um ohne Nachfrage zu löschen.
    """
    enable_utf8_console()

    engine = get_engine(host="localhost")
    Session = sessionmaker(bind=engine)
    session = Session()

    if confirm:
        user_input = input("⚠️  Bist du sicher, dass du ALLE Tabellen löschen willst? (yes/no): ")
        if user_input.lower() != "yes":
            print("❌ Abgebrochen.")
            session.close()
            return

    print("🧹 Lösche alle Tabellen ...")
    Base.metadata.drop_all(engine)
    print("✅ Alle Tabellen gelöscht.")

    print("🛠️  Erstelle Tabellen neu ...")
    create_tables(engine)
    print("✅ Tabellen neu erstellt.")

    session.close()
    print("🎉 Datenbank wurde erfolgreich gewiped und neu initialisiert.")


if __name__ == "__main__":
    wipe_database()