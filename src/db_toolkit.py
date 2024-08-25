from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai import AzureChatOpenAI
from env import load_env

load_env()


def main():
    db = SQLDatabase.from_uri("sqlite:////Users/francesco.campanile/DataGripProjects/testing-langchain/identifier.sqlite")
    initialize_llm_agent(db)


def initialize_llm_agent(db):
    llm = AzureChatOpenAI(deployment_name="GPT-4-Turbo")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(llm, toolkit=toolkit, agent_type="openai-tools", verbose=True,
                                      handle_parsing_errors=True)
    print(agent_executor.invoke(
        "Can you retrieve all the movies where John Belushi was an actor, ensuring to skip any duplicates, and then provide details about the genres of the three highest-rated ones?"
    ))


if __name__ == "__main__":
    main()
