# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

import os
import sqlite3
from sqlite3 import Error


import pandas as pd

TABLE_NAME = "LONDON"

CSV_COL_NAMES = ["Month", "Latitude", "Longitude", "Location", "Crime type"]
DB_COL_NAMES = ["MONTH", "LATITUDE", "LONGITUDE", "DESCRIPTION", "CRIME_TYPE"]
DB_COL_TYPES = ["TEXT", "REAL", "REAL", "TEXT", "TEXT"]


def list_files(startpath):
    full_file_paths = []
    for root, directories, filenames in os.walk(startpath):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_file_paths.append(os.path.join(root, filename))
    return full_file_paths


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
        return conn
    except Error as e:
        print(e)


def create_crime_table(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("CREATE TABLE {tn} (ID INTEGER)".format(tn=TABLE_NAME))

    for (col_name, col_type) in zip(DB_COL_NAMES, DB_COL_TYPES):
        cursor.execute("ALTER TABLE {tn} ADD COLUMN {cn} {ct}".format(tn=TABLE_NAME, cn=col_name, ct=col_type))


def move_csv_to_sql(sql_conn, csv_file_paths):
    for file_path in csv_file_paths:
        print(file_path)
        crime_df = pd.read_csv(file_path, usecols=CSV_COL_NAMES)
        crime_df.columns = DB_COL_NAMES
        print(crime_df.shape)
        crime_df.to_sql(TABLE_NAME, sql_conn, if_exists='append', index_label="ID")


def get_specific_crimes(db_conn, crime_type, start_date, end_date):
    query = """
        SELECT * FROM {tn}
        WHERE {tn}.CRIME_TYPE='{ct}' AND LATITUDE IS NOT NULL     
    """.format(tn=TABLE_NAME, ct=crime_type)
    crime_all_period = pd.read_sql(query, db_conn, parse_dates=["MONTH"], index_col="ID")
    return crime_all_period[(crime_all_period['MONTH'] >= start_date) & (crime_all_period['MONTH'] <= end_date)]


def bootstrap_scripts(input_filepath, output_filepath):
    db_conn = create_connection(output_filepath)
    try:
        create_crime_table(db_conn)
    except sqlite3.OperationalError as error:
        print(error)

    all_crime_csv_file_paths = list_files(input_filepath)
    move_csv_to_sql(db_conn, all_crime_csv_file_paths)
    db_conn.close()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    if os.path.isfile(output_filepath):
        logger.info("Removing previous database.")
        os.remove(output_filepath)

    bootstrap_scripts(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
