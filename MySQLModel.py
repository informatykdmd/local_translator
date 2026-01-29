import pymysql
import re
from bin.config_utils import DBDATA as DB  # Import konfiguracji połączenia
from bin.appslib import handle_error_Turbo

class MySQLModel:
    _global_conn = None  # Globalne połączenie dla stałego trybu

    def __init__(self, permanent_connection=True):
        """
        Tworzy dynamiczny model dla dowolnej tabeli w bazie danych MySQL.

        :param permanent_connection: Czy połączenie ma być stałe (domyślnie True)
        """
        self.permanent_connection = permanent_connection
        self.columns = []  # Lista kolumn będzie pobierana dynamicznie

        # Sprawdzenie, czy połączenie ma być stałe
        if self.permanent_connection:
            if MySQLModel._global_conn is None:
                MySQLModel._global_conn = self._connect_db()
            self.conn = MySQLModel._global_conn
        else:
            self.conn = self._connect_db()

        self.cursor = self.conn.cursor()

    def _connect_db(self):
        """ Łączy się z bazą danych na podstawie danych z config_utils """
        return pymysql.connect(
            user=DB['user'],
            password=DB['pass'],
            host=DB['host'],
            database=DB['base'],
            cursorclass=pymysql.cursors.DictCursor
        )

    def _fetch_columns(self, table_name):
        """ Pobiera nazwy kolumn dla danej tabeli (wywoływane dynamicznie) """
        self.cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        self.columns = [col["Field"] for col in self.cursor.fetchall()]
        for col in self.columns:
            setattr(self, col, None)  # Domyślnie ustawiamy None dla każdej kolumny

    @staticmethod
    def _extract_table_name(query):
        """ Analizuje zapytanie SQL i wyciąga nazwę tabeli z `FROM`, `JOIN`, `UPDATE`, `INSERT INTO` """
        try:
            # Czyszczenie białych znaków i normalizacja spacji
            query = re.sub(r'\s+', ' ', query.strip())  # Usuwa nadmiarowe spacje, tabulatory, nowe linie

            # Wyszukiwanie w SQL (ignorujemy wielkość liter tylko w regex, ale nie zmieniamy oryginału)
            match = re.search(r'\b(from|join|update|into)\s+([a-zA-Z0-9_]+)', query, re.IGNORECASE)
            if match:
                return match.group(2)  # Nazwa tabeli

        except Exception as e:
            print(f"Nie udało się rozpoznać tabeli: {e}")
            handle_error_Turbo(f"Nie udało się rozpoznać tabeli: {e}")

        return None

    def set_values(self, data):
        """ Przypisuje wartości do atrybutów obiektu na podstawie danych z bazy """
        for col, value in data.items():
            setattr(self, col, value)

    def fetch_one(self, query, params=None):
        """ Pobiera pierwszy rekord i przypisuje wartości do obiektu """
        table_name = self._extract_table_name(query)  # Poprawne wywołanie metody statycznej
        if table_name:
            self._fetch_columns(table_name)

        self.cursor.execute(query, params)
        row = self.cursor.fetchone()
        if row:
            self.set_values(row)

    def executeTo(self, query, params=None):
        """ Wykonuje zapytania, które nie zwracają danych (INSERT, UPDATE, DELETE) """
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"Błąd SQL: {e}")
            handle_error_Turbo(f"Błąd SQL: {e}")
            return False

    def getFrom(self, query, params=None, as_dict=False, as_object=False):
        """
        Wykonuje zapytanie SELECT i zwraca wyniki w różnych formatach.

        :param query: Zapytanie SQL
        :param params: Parametry do zapytania (opcjonalne)
        :param as_dict: Jeśli True, zwraca listę słowników
        :param as_object: Jeśli True, zwraca listę obiektów MySQLModel
        :UWAGA! Parametry as_dict i as_object nie mogą być jednocześnie True.
        :return: Wynik w formacie tuple, dict lub obiektowym
        """
        if as_dict and as_object:
            handle_error_Turbo(f"Błąd: as_dict i as_object nie mogą być jednocześnie ustawione na True!")
            raise ValueError("Parametry as_dict i as_object nie mogą być jednocześnie True. Wybierz jedno.")


        table_name = self._extract_table_name(query)

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        if table_name:
            self._fetch_columns(table_name)

        if as_dict:
            return rows  # Zwraca listę słowników
        
        if as_object:
            objects = []
            for row in rows:
                obj = MySQLModel(self.permanent_connection)
                obj.set_values(row)
                objects.append(obj)
            return objects
        
        return [tuple(row.values()) for row in rows]  # Domyślnie zwraca jako tuple

    def close_connection(self):
        """ Zamknięcie połączenia z bazą (jeśli nie jest permanentne) """
        if not self.permanent_connection:
            self.cursor.close()
            self.conn.close()
        elif MySQLModel._global_conn:
            MySQLModel._global_conn.close()
            MySQLModel._global_conn = None

    def __repr__(self):
        """ Reprezentacja tekstowa obiektu """
        attributes = {k: v for k, v in self.__dict__.items() if k not in ['conn', 'cursor']}
        return f"<{self.__class__.__name__} {attributes}>"

# 🔹 Przykłady użycia
if __name__ == "__main__":
    # 🌟 Stałe połączenie (domyślnie)
    db = MySQLModel(permanent_connection=True)

    # Pobranie użytkownika (SQL sam wykryje tabelę!)
    db.fetch_one("SELECT * FROM contact WHERE id=11356")
    print(db)

    # Pobranie produktów (SQL sam wykryje tabelę!)
    db.fetch_one("SELECT * FROM contact WHERE id=3")
    print(db)

    # Pobranie listy użytkowników jako dict
    print(db.getFrom("SELECT * FROM contact LIMIT 5", as_dict=True))

    # Pobranie listy zamówień jako lista obiektów
    orders = db.getFrom("SELECT * FROM contact LIMIT 5", as_object=True)
    for order in orders:
        print(order)

    # # Wykonanie INSERT bez podawania tabeli
    # db.executeTo("INSERT INTO xusers (name, email) VALUES (%s, %s)", ("Nowy", "nowy@example.com"))

    # Zamknięcie połączenia
    db.close_connection()
