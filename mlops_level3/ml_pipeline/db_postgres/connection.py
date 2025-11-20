import psycopg2
from psycopg2.extras import RealDictCursor

DB_HOST = "localhost"
DB_NAME = ["raw_db", "cleaned_db"]  # Switch between raw_db and clean_db
DB_USER = "mlops_user"
DB_PASSWORD = "testing321"
DB_PORT = 5432  # Standard Postgres Port


def get_connection(db_name=DB_NAME):
    """
    Return a new connection to the PostgreSQL database.
    (psycopg2 connection object)
    """
    return psycopg2.connect(
        host=DB_HOST,
        database=db_name,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        cursor_factory=RealDictCursor # Return results as dictionaries
    )