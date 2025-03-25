import os
import pwd
import mimetypes

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

from typing import Literal
from typing_extensions import Annotated
import autogen

import ollama

def getLlamaResponse(prompt, context):
    user_message = f'''I want to find out exact information.
    Here is the question {prompt}.

    To answer this question, I have data available from a tool function from where I need to retreive answer.
    Here is the data: {context}
    '''
    return ollama.chat(model="llama3.2",
                       messages=[{
                           "role": "user",
                           "content": user_message
                           }],
                       )

# model_client = OpenAIChatCompletionClient(model="deepseek-r1",
#                                           base_url="http://0.0.0.0:4000",
#                                           api_key="placeholder",
#                                           model_info={
#                                               "model": "deepseek-r1",
#                                               "vision": False,
#                                               "function_calling": True,
#                                               "json_output": False,
#                                               "family": ModelFamily.R1
#                                           },
#                                           )
# define tools
def get_FileMetaData(file_path: str) -> str:
    # Get file size
    file_size = os.path.getsize(file_path)
    # Get file owner
    file_owner = pwd.getpwuid(os.stat(file_path).st_uid).pw_name
    # Get file type
    file_types, _ = mimetypes.guess_type(file_path)
    return f"The file path is {file_path}, file owner is {file_owner}, file size is {file_size}, file type is {file_types}"

local_llm_config = {
    "config_list": [
        {
            "model": "NotRequired",     # Loaded with LiteLLM command
            "api_key": "NotRequired",       # Not needed
            "base_url": "http://0.0.0.0:4000",  # Your LiteLLM URL
            "price": [0, 0],    # Put in price per 1K tokens [prompt, response] as free!
        }
    ],
    "cache_seed": None,
}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="""For file management prompts,
    only use the functions you have been provided with.
    If the function has been called previously,
    return oly the word 'TERMINATE'.""",
    llm_config=local_llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config={"work_dir": "code", "use_docker": False},
)

# Register the function with the agent
@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="file metadata function")
def get_Metadata(
    file_path: str,
) -> str:
    return get_FileMetaData(file_path)


prompt = "find the file size of a file name samsung_stock_price.png"
async def main() -> None:
    res = user_proxy.initiate_chat(
        chatbot,
        message=prompt,
        # summary_method="reflection_with_llm",
    )

    print(f"chat messages: {chatbot.chat_messages}")

    # format your response
    print(getLlamaResponse(prompt, chatbot.chat_messages))

    # agent = AssistantAgent(name="Agent_Filer",
    #                        model_client=model_client,
    #                        # tools = [tools]
    #                        )
    # print(await agent.run(task="Say 'Hello World!'"))


asyncio.run(main())

