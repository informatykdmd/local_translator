from datetime import datetime
import os
import logging

def handle_error(exception, retry_count=3, log_path="../logs/errors.log"):
    try:
        with open(log_path, "a") as log:
            now = str(datetime.now())
            message = "{0} {1}\n".format(now, exception)
            log.write(message)
    except Exception as e:
        if retry_count > 0:
            print(f"Błąd podczas zapisywania do pliku: {e}. Ponawiam próbę...")
            handle_error(exception, retry_count - 1, log_path)
        else:
            print("Nieudana próba zapisu do pliku. Przekroczono limit ponawiania.")


# Ustalanie ścieżki logów (project/logs/errors.log)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, "logs")
log_file_path = os.path.join(log_dir, "errors.log")

# Tworzenie katalogu logs jeśli nie istnieje
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Konfiguracja loggera - ustawiana raz
logger = logging.getLogger("AppLogger")
if not logger.hasHandlers():  # <--- kluczowe, żeby unikać duplikacji handlerów
    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Handler do pliku
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler do konsoli (opcjonalnie)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Funkcja logująca
def handle_error_Turbo(exception, retry_count=3, log_path=None):
    if log_path is None:
        log_path = log_file_path  # Zawsze logujemy do tego samego pliku

    logger.error(exception)  # Log do pliku i konsoli

    try:
        with open(log_path, "a", encoding="utf-8") as log:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{now} {exception}\n"
            log.write(message)
    except Exception as e:
        if retry_count > 0:
            logger.error(f"Błąd podczas zapisywania do pliku: {e}. Ponawiam próbę...")
            handle_error_Turbo(exception, retry_count - 1, log_path)
        else:
            logger.error("Nieudana próba zapisu do pliku. Przekroczono limit ponawiania.")

