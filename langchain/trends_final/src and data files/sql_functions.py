import sqlite3
import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# SQLite connection setup
def connect_db(db_path):
    # Ensure the directory exists
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        print(f"Directory {db_dir} does not exist. Creating it.")
        os.makedirs(db_dir)  # Create directory if it doesn't exist

    # Check if the file exists and is accessible
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist. Creating the file.")

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        print(f"Successfully connected to the database at {db_path}")
        return conn
    except sqlite3.OperationalError as e:
        print(f"Error while connecting to the database: {e}")
        raise

def get_table_info(conn):
    """
    Retrieve all table names and sample rows from the database.
    """
    try:
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("No tables found in the database.")
            return None, None

        table_names = [table[0] for table in tables]

        # Get DDL and sample rows for each table
        table_info = []
        for table in table_names:
            cursor.execute(f"PRAGMA table_info({table});")
            ddl = cursor.fetchall()

            cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
            sample_rows = cursor.fetchall()

            table_info.append((table, ddl, sample_rows))

        return table_names, table_info
    except sqlite3.Error as e:
        print(f"Error retrieving table information: {e}")
        return None, None

def llm_create_sql(table_info, question, lang_model):
    """
    Use the LLM to generate a SQL query based on table information and the user question.
    """
    # Format table information to include only names and DDL (excluding "Sample Rows")
    tables_summary = "\n\n".join(
        f"Table Name: {table}\nSchema: {[(col[1], col[2]) for col in ddl]}" for table, ddl, rows in table_info
    )

    # Updated prompt with clear instructions and clean metadata
    create_prompt = PromptTemplate(
        input_variables=["tables_summary", "question"],
        template=f"""
        The following is the schema of tables in the database:
        {tables_summary}

        Using valid SQL syntax, answer the following question:
        {question}
        """
    )

    # Initialize the LLMChain with the prompt and the LLM model
    create_chain = LLMChain(llm=lang_model, prompt=create_prompt, verbose=False)

    # Generate SQL query
    sql_query = create_chain.predict(tables_summary=tables_summary, question=question)
    sql_query = "SELECT "+sql_query.split("SELECT", 1)[-1].strip()
    print("sql-query: ", sql_query)
    return sql_query
