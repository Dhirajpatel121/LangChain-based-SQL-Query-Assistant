import os
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import streamlit as st
import pandas as pd  # For result handling
from sql_functions import connect_db, get_table_info, llm_create_sql
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Initialize Local LLM
model_id = 'NumbersStation/nsql-350M'  # Replace with your local model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=False)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=10000,
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# Database filepaths
databases = {
    "Chinook": {
        "path": r"C:\Users\mehul\OneDrive\Documents\Trends Marketplace\trendsRepo\text-to-sql\src\app\Chinook_Sqlite.sqlite",  # Replace with your Chinook database path
        "erd": "chinook_erd.png"
    },
    "E-commerce": {
        "path": r"C:\Users\mehul\OneDrive\Documents\Trends Marketplace\trendsRepo\text-to-sql\src\app\olist.sqlite",  # Replace with your E-commerce database path
        "erd": "ecommerce_erd.png"
    },
    "Pets-stackexchange": {
        "path": r"C:\Users\mehul\OneDrive\Documents\Trends Marketplace\trendsRepo\text-to-sql\src\app\pets_stackexchange.sqlite",  # Replace with your E-commerce database path
        "erd": "ecommerce_erd.png"
    }
}

def sql_copilot(language_model=local_llm):
    st.title("langChain Based SQL Assistant")
    st.markdown("### LLM-Powered SQL Assistant")

    # Database selection
    db_choice = st.selectbox("Select the database", list(databases.keys()))
    db_filepath = databases[db_choice]["path"]
    erd_image = databases[db_choice]["erd"]

    # Input query
    user_question = st.text_input("Enter your question here", "")

    # Initialize tabs
    tabs = st.tabs(["Result", "Query", "ERD"])

    # Check if query is provided
    if user_question:
        # Connect to the selected database
        conn = connect_db(db_filepath)
        if not conn:
            st.error("Failed to connect to the database.")
            return

        # Retrieve table info
        table_names, table_info = get_table_info(conn)
        if not table_info:
            st.error("No valid tables found in the database.")
            return

        # Generate SQL query
        with tabs[1]:  # "Query" tab
            st.markdown("#### Generated SQL Query")
            try:
                sql_query = llm_create_sql(table_info=table_info, question=user_question, lang_model=language_model)
                # Display the query as a readable block without a scroller
                st.text_area("SQL Query", value=sql_query, height=200, max_chars=None)
            except Exception as e:
                st.error(f"Error generating SQL query: {e}")
                return

        # Execute the query and display results
        with tabs[0]:  # "Result" tab
            st.markdown("#### Query Results")
            try:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                result = cursor.fetchall()

                # Fetch column names
                columns = [desc[0] for desc in cursor.description]

                # Convert to a pandas DataFrame
                result_df = pd.DataFrame(result, columns=columns)

                if not result_df.empty:
                    st.dataframe(result_df)  # Display results as a table
                    st.download_button(
                        label="Download Results as CSV",
                        data=result_df.to_csv(index=False),
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Query executed successfully but returned no results.")
            except sqlite3.Error as e:
                st.error(f"Error executing SQL query: {e}")
            finally:
                conn.close()

        # ERD visualization
        with tabs[2]:  # "ERD" tab
            st.markdown("#### Entity-Relationship Diagram (ERD)")
            st.image(erd_image, caption=f"{db_choice} Database ERD")

if __name__ == '__main__':
    sql_copilot()
