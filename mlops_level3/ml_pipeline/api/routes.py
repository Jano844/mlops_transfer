from fastapi import APIRouter, UploadFile, File
from db_postgres.connection import get_connection
import pandas as pd

router = APIRouter()

@router.get("/ping")
def ping():
    """Health check endpoint"""
    return {"message": "pong"}



@router.get("/test")
def health_check(test: str):
    return {"test": "{}".format(test)}






def check_input_data(data: pd.DataFrame):
    expected_columns = \
        ["ffid", "height", "loaded", "onduty", "timeStamp", "latitude", "longitude", "speed"]

    if list(data.columns) != expected_columns:
        return 1, {"error": "CSV columns do not match the required format  {}".format(expected_columns)}
    
    if len(data) == 0:
        return 1, {"error": "CSV file is empty"}
    return 0, {"message": "Input data is valid"}


def insert_data_to_db(data: pd.DataFrame, db_conn):
    cursor = db_conn.cursor()
    for _, row in data.iterrows():
        try:
            cursor.execute(
                """
                INSERT INTO vehicle_data (ffid, height, loaded, onduty, timestamp, latitude, longitude, speed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(row['ffid']),
                    str(row['height']),
                    str(row['loaded']),
                    str(row['onduty']),
                    str(row['timeStamp']),
                    str(row['latitude']),
                    str(row['longitude']),
                    str(row['speed'])
                )
            )
        except Exception as e:
            print(f"Error inserting row {row}: {e}")
            db_conn.rollback()
    db_conn.commit()
    cursor.close()
    db_conn.close()

@router.post("/insert_data")
async def input_data(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        return {"error": "only CSV files are allowed"}
    data = pd.read_csv(file.file)
    error, response = check_input_data(data)
    if error:
        return response

    db_conn = get_connection(db_name="raw_db")
    if not db_conn:
        return {"error": "Failed to connect to the database"}
    
    insert_data_to_db(data, db_conn)
    return {"message": "Data inserted successfully"}
