import requests
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import kagglehub


def obtener_tickers_sp500():
    url = "https://financhle.com/sp500-companies-by-weight#SP500CompleteList"
    
    # Hacer la solicitud HTTP
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontrar las tabla de tickers
    tables = soup.find_all('table')

    table  = tables[-1]
    
    # Leer la tabla en un DataFrame de pandas
    df = pd.read_html(str(table))[0]
    return df

def get_columns(df):
    return ', '.join(df.columns)
    

if __name__ == "__main__":
    df = obtener_tickers_sp500()


    # Download latest version
    path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
    companies_path = path+"/sp500_companies.csv"

    
    df2 = pd.read_csv(companies_path)
    
    df = pd.merge(df, df2, left_on='Ticker', right_on='Symbol')
    df = df.drop(columns=['Longbusinesssummary', 'Exchange', 'Shortname', 'Weight','Currentprice'])

    print(df.head())
    

    columns = get_columns(df)
    #print(columns)
    
    df_string = df.to_string(index=False)
    #print(df_string)
    #exit(0)

    prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                f"""Eres un asistente financiero que provee información acerca de las acciones del S&P 500
                    Tienes un acceso a un DataFrame con las siguientes columnas
    
                    {columns}

                Contenido del dataframe 

                {df_string}

            """
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
    
    model = ChatOpenAI(model="gpt-4o-2024-08-06")

# Define the function that calls the model
    def call_model(state: MessagesState):
        chain = prompt | model
        response = chain.invoke(state)
        return {"messages": response}
    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        input_messages = [HumanMessage(user_input)]
        output = app.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()  # output contains all messages in state