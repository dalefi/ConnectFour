import subprocess
import os
from datetime import datetime

# Konfiguration aus deiner docker-compose.yml
SERVICE_NAME = "postgres"  # Der Name unter 'services:'
DB_USER = "daniel"
DB_PASS = "connectfour"
DB_NAME = "connectfour"
BACKUP_DIR = "backups"


def create_backup():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{BACKUP_DIR}/{DB_NAME}_backup_{timestamp}.bak"

    # docker compose exec nutzt den Service-Namen aus deiner YAML
    command = [
        "docker", "compose", "exec", "-e", f"PGPASSWORD={DB_PASS}",
        SERVICE_NAME,
        "pg_dump",
        "-U", DB_USER,
        "-F", "c",
        DB_NAME
    ]

    try:
        print(f"Erstelle Backup für Service '{SERVICE_NAME}'...")
        with open(filename, "wb") as f:
            # stdout wird in die Datei geschrieben
            subprocess.run(command, check=True, stdout=f)

        print(f"Erfolgreich gespeichert: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Fehler: Stell sicher, dass 'docker compose up' läuft.\n{e}")


if __name__ == "__main__":
    create_backup()