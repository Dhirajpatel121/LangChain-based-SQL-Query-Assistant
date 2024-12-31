import os
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import streamlit as st
import pandas as pd  # For result handling
from sql_functions import connect_db, get_table_info, llm_create_sql
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #E8F5E9;
        }

        .main {
            background: #121212;
            color: #E8F5E9;
        }

        .header {
            display: flex;
            align-items: center;
            padding: 10px 20px;
            background-color: #00471E;
            color: #FFFFFF;
        }

        .header img {
            width: 50px;
            margin-right: 15px;
        }

        .header h1 {
            margin: 0;
            font-size: 2rem;
        }

        .stButton > button {
            background-color: #00471E !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 15px !important;
            font-weight: bold !important;
        }

        .stTextInput > div > input {
            background-color: #1E1E1E !important;
            color: white !important;
            border: 1px solid #00471E !important;
            border-radius: 8px !important;
        }

        .database-description {
            font-size: 1rem;
            margin: 15px 0;
            color: #E0E0E0;
        }
    </style>
""", unsafe_allow_html=True)

# Add University Logo and Header with Picture on Extreme Right
st.markdown(f"""
    <div class="header" style="display: flex; align-items: center; justify-content: space-between; padding: 10px 20px; background-color: #00471E; color: #FFFFFF;">
        <div style="flex-grow: 1;">
            <h1 style="margin: 0; font-size: 2rem;">LLM4SQL: LangChain Powered SQL Assistant</h1>
        </div>
    </div>
""", unsafe_allow_html=True)


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

# Database filepaths and descriptions
databases = {
    "Chinook": {
        "path": r"C:\Users\dhiru\Downloads\trends_final\src\Chinook_Sqlite.sqlite",
        "erd": "chinook_erd.png",
        "description": "The Chinook database contains a digital media store schema, including tables for customers, invoices, and tracks."
    },
    "E-commerce": {
        "path": r"C:\Users\dhiru\Downloads\trends_final\src\olist.sqlite",
        "erd": "ecommerce_erd.png",
        "description": "The E-commerce database includes information about customers, orders, and products for an online marketplace."
    },
    "Employees": {
        "path": r"C:\Users\dhiru\Downloads\trends_final\src\company_employee.sqlite",
        "erd": "company employee erd.png",
        "description": "The Company employee database contains details related to the company and employees."
    }
}

def sql_copilot(language_model=local_llm):
    # Database Selection
    st.markdown("### Select a Database")
    db_choice = st.selectbox("Select the database below", list(databases.keys()))
    db_filepath = databases[db_choice]["path"]
    erd_image = databases[db_choice]["erd"]
    db_description = databases[db_choice]["description"]

    # Display database description
    st.markdown(f"<div class='database-description'>{db_description}</div>", unsafe_allow_html=True)

    # Display ERD Button
    if st.button("View Entity-Relationship Diagram (ERD)"):
        st.image(erd_image, caption=f"{db_choice} Database ERD")

    # Input for User Query
    st.markdown("### Enter Your Question")
    user_question = st.text_input("Type your question here", "")

    # Tabs for Results and Query
    tabs = st.tabs(["Result", "Generated Query"])

    if user_question:
        # Connect to the database
        conn = connect_db(db_filepath)
        if not conn:
            st.error("Failed to connect to the database.")
            return

        # Get table information
        table_names, table_info = get_table_info(conn)
        if not table_info:
            st.error("No valid tables found in the database.")
            return

        with tabs[1]:
            st.markdown("#### Generated SQL Query")
            try:
                sql_query = llm_create_sql(table_info=table_info, question=user_question, lang_model=language_model)
                st.text_area("SQL Query", value=sql_query, height=200)
            except Exception as e:
                st.error(f"Error generating SQL query: {e}")
                return

        with tabs[0]:
            st.markdown("#### Query Results")
            try:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                result_df = pd.DataFrame(result, columns=columns)

                if not result_df.empty:
                    st.dataframe(result_df)
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

if __name__ == '__main__':
    sql_copilot()
