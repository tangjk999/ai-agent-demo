from pydantic_ai import Agent
from pydantic_ai import DeepSeekModel

from dotenv import load_dotenv
import tools
import os
import getpass

load_dotenv()
model = DeepSeekModel("deepseek-chat", api_key=getpass.getpass("请输入API密钥: "))

agent = Agent(model,
              system_prompt="You are an experienced programmer",
              tools=[tools.read_file, tools.list_files, tools.rename_file])

def main():
    history = []
    while True:
        user_input = input("Input: ")
        resp = agent.run_sync(user_input,
                              message_history=history)
        history = list(resp.all_messages())
        print(resp.output)


if __name__ == "__main__":
    main()
