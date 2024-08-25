import yaml
from langchain.agents import create_json_agent
from langchain_community.agent_toolkits import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain_openai import AzureChatOpenAI
from env import load_env

load_env()


def main():
    with open("../openai_openapi.yml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    json_spec = JsonSpec(dict_=data, max_value_length=200)
    json_toolkit = JsonToolkit(spec=json_spec)

    json_agent_executor = create_json_agent(
        llm=AzureChatOpenAI(deployment_name="GPT-4-Turbo"), toolkit=json_toolkit, verbose=True
    )

    print(json_agent_executor.invoke(
        "What are the possible negative cases when I call an API to update a pet info? I don't recall the endpoint name, but it should be a PUT verb"
    ))


if __name__ == "__main__":
    main()
