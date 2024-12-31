import os
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import streamlit as st
import pandas as pd
from sql_functions import connect_db, get_table_info, llm_create_sql
import warnings
import base64

warnings.filterwarnings("ignore", category=UserWarning)

# Convert images to Base64
def convert_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Updated file paths
# db_icon_base64 = convert_image_to_base64(r"C:\Users\mehul\OneDrive\Desktop\trends_final\src\data-server.png")
#fish_to_db_base64 = convert_image_to_base64(r"C:\Users\mehul\OneDrive\Desktop\trends_final\src\Langchain--Streamline-Simple-Icons.svg")
# arrow_icon_base64 = convert_image_to_base64(r"C:\Users\mehul\OneDrive\Desktop\trends_final\src\exchange.png")

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
            max-width: 100%;
            padding: 0px 20px;
        }

        .icons {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            gap: 20px;
        }

        .icon {
            height: 60px;
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px 0;
            background-color: #00471E;
            color: #FFFFFF;
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


# Add Header Below the Icons
st.markdown("""
    <div class="header">
        <h1>LLM4SQL: LangChain Powered SQL Assistant</h1>
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

# Define the common base path
base_path = r"C:\Users\dhiru\Downloads\trends_final\src"

# Database filepaths and descriptions
databases = {
    "Chinook": {
        "path": os.path.join(base_path, "Chinook_Sqlite.sqlite"),
        "erd": "chinook_erd.png",
        "description": "The Chinook database contains a digital media store schema, including tables for customers, invoices, and tracks."
    },
    "E-commerce": {
        "path": os.path.join(base_path, "olist.sqlite"),
        "erd": "ecommerce_erd.png",
        "description": "The E-commerce database includes information about customers, orders, and products for an online marketplace."
    },
    "Employee DB": {
        "path": os.path.join(base_path, "company_employee.sqlite"),
        "erd": "company_employee.png",
        "description": "The Company employee database contains details related to the company and employees."
    }
}

def sql_copilot(language_model=local_llm):
    # Database Selection
    st.markdown("### Select a Database")
    db_choice = st.selectbox("Select the database", list(databases.keys()))
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

    # Tabs for Query and Results
    tabs = st.tabs(["Generated Query", "Result"])

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

        with tabs[0]:
            st.markdown("#### Generated SQL Query")
            try:
                sql_query = llm_create_sql(table_info=table_info, question=user_question, lang_model=language_model)
                st.text_area("SQL Query", value=sql_query, height=200)
            except Exception as e:
                st.error(f"Error generating SQL query: {e}")
                return

        with tabs[1]:
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
                st.error("Oops, the model requires more nuanced training....")
            finally:
                conn.close()

if __name__ == '__main__':
    sql_copilot()
