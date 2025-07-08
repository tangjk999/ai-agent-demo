import os
from dotenv import load_dotenv
from deepseek_llm import DeepSeekLLM

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") or input("请输入DeepSeek API密钥: ")

llm = DeepSeekLLM(api_key=api_key)

def main():
    history = []
    print("输入 exit 退出")
    while True:
        user_input = input("Input: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            print("输入不能为空，请重新输入。")
            continue
        # 构造历史对话
        prompt = ""
        for msg in history:
            if msg["role"] == "user":
                prompt += f"用户：{msg['content']}\n"
            else:
                prompt += f"助手：{msg['content']}\n"
        prompt += f"用户：{user_input}\n助手："
        reply = llm.invoke(prompt)
        print("DeepSeek:", reply)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
