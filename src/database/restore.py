import subprocess
import os

SERVICE_NAME = "postgres"
DB_USER = "daniel"
DB_PASS = "connectfour"
DB_NAME = "connectfour"


def restore_backup(backup_file):
    if not os.path.exists(backup_file):
        print(f"Datei nicht gefunden: {backup_file}")
        return

    command = [
        "docker", "compose", "exec", "-T",
        "-e", f"PGPASSWORD={DB_PASS}",
        SERVICE_NAME,
        "pg_restore",
        "-U", DB_USER,
        "-d", DB_NAME,
        "--data-only"
    ]

    try:
        print(f"Stelle Backup {backup_file} wieder her...")
        with open(backup_file, "rb") as f:
            # Datei wird per stdin in den Container gestreamt
            subprocess.run(command, check=True, stdin=f)
        print("Wiederherstellung erfolgreich abgeschlossen!")
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei der Wiederherstellung: {e}")


if __name__ == "__main__":
    # Listet verfügbare Backups auf
    if os.path.exists("backups"):
        print("Verfügbare Backups:")
        for f in os.listdir("backups"):
            print(f" - backups/{f}")

    file_path = input("\nPfad zum Backup eingeben: ")
    restore_backup(file_path)