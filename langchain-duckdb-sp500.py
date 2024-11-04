import os
import requests
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_openai import ChatOpenAI
import kagglehub
import duckdb
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#os.environ( "USER_AGENT","Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")


class DuckDBQuery(BaseModel):
    sql_query:str = Field(description="Query to be executed")
    can_generate_query : bool = Field(description="Is it possible to generate a query")
    result_explain: str = Field(description="Explicación del resultado del query y lo que significa")


def download_sp500_from_financhle():
    url = "https://financhle.com/sp500-companies-by-weight"
    
    # Hacer la solicitud HTTP
    response = requests.get(url)
    print(response.text)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontrar las tabla de tickers
    tables = soup.find_all('table')

    table  = tables[-1]
    
    # Leer la tabla en un DataFrame de pandas
    df = pd.read_html(str(table))[0]


    df['% of S&P 500'] = df['% of S&P 500'].str.replace('%', '').astype(float)
    df["Today's Change (%)"] = df["Today's Change (%)"].str.replace('%', '').astype(float)

    # Remove dollar signs and convert to floats
    df['Price'] = df['Price'].str.replace('[$,]', '', regex=True).astype(float)
    df["Today's Change ($)"] = df["Today's Change ($)"].str.replace('[$,]', '', regex=True).astype(float)
    return df


def download_sp500_from_kaggle():
    # Download latest version
    path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
    companies_path = path+"/sp500_companies.csv"


    df = pd.read_csv(companies_path)
    return df


def get_columns(df):
    return ', '.join(df.columns)


def call_model(state: MessagesState):
        chain = prompt | model
        response = chain.invoke(state)
        return {"messages": response}

if __name__ == "__main__":
    df_web    = download_sp500_from_financhle()
    df_kaggle = download_sp500_from_kaggle()

    
    df_sp500 = pd.merge(df_web, df_kaggle, left_on='Ticker', right_on='Symbol')
    df_sp500.to_excel("data/sp500.xlsx",index=False)

    
  
            

    df_string = df_sp500.to_string(index=False)
    columns = get_columns(df_sp500)
    prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
f"""Eres un asistente financiero que provee información acerca de las acciones del S&P 500
Tienes un acceso a un DataFrame con las siguientes columnas. 
    {columns}

La información que obtienes del dataframe es reciente y puedes confiar en esta.
"""
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
    
    model = ChatOpenAI(model="gpt-4o-2024-08-06")
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}

    while True:

        model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
        structured_llm = model.with_structured_output(DuckDBQuery)
        columns = get_columns(df_sp500)
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        question = user_input
        resp_duck : DuckDBQuery = structured_llm.invoke(
f"""Eres un experto en análisis de datos y sabes usar duckdb.
Puedes usar conocimientos de cultura general para completar el query. 
Trata de usar el Ticker preferentemente al nombre de la empresa.
Incluye en el query el ticker, nombre de la empresa y otras columnas que enriquezcan la respuesta que será usada por un LLM.

Tienes los datos del SP500 en el dataframe df_sp500 con las siguientes columnas {columns}

genera un query de duckdb para resolver las preguntas. 

{question}
 
""")
        
        input_messages = [HumanMessage(user_input)]
        if resp_duck.can_generate_query:
            print("SQL", resp_duck.sql_query)
            df_duck = duckdb.query(resp_duck.sql_query).to_df()
            print(df_duck)
            ai_message = "He hecho el siguiente análisis con información reciente. "+resp_duck.result_explain+"\n el resultado es:\n"+df_duck.to_string()
            print("ai_message",ai_message)
            print("------------------------------------")
            input_messages = [AIMessage(content=ai_message),HumanMessage(user_input)]
        output = app.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()  # output contains all messages in state
        #print(output["messages"][-1].pretty_print())


