import autogen
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
import os

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
model =  os .getenv("OPENAI_MODEL")


config_list = [{'model':model, 'api_key':apikey}]

llm_config = {"config_list": config_list}


task = '''
    **Task**: As an architect, you are required to design a solution for the following business requirements:
        - Data storage for massive amounts of IoT data
        - Real time data analytics and machine learning pipeline
        - Scalability
        - Cost Optimization
        - Region pairs in Europe, for disaster recovery
        - Tools for monitering and observability
        - Timeline: 6 months

        Break the problem using Chain of Thought approch. 
        Ensure that your solutiona architecture is following best practices. 
'''

cloud_prompt = '''
            **Role**: You are an expert cloud architect. 
            You need to develop architecture proposals using either cloud -specific PaaS service, or cloud-agnostic ones.
            The final proposal should consider all 3 main cloud providers: Azure, AWS, and GCP, and provide a data architecture for each. 
            At the end, briefly state the advantages of cloud over on-premises architecures, and summerize your solutions for each cloud provider using table of clarity.
'''

cloud_prompt = cloud_prompt + task



oss_prompt = '''
           **Role**: You are an expert on-premises, open-source software architect. 
           You need to develop architrcture proposals without considering cloud solutions.
           Only  use open-source frameworks that are popular hand have losts of active contributors.
           At the end, briefly state the advantages of the open-source adoptions, and summerize your solution using a table of clarity.
'''

oss_prompt = oss_prompt + task


lead_prompt = '''
        **Role**: You are a lead Architect tasked with managing a conversation between the cloud and the open-source Architects.
        Each Architect will perform a task and respond with their results. 
        You will crictically review those and also ask for, or point to, the disadvantages of their solutions.
        You will review each results, and choose the best solution in accordance with the business requirements and architecure best practices. 
        You will use any number of the summary tables to communicate your decisions.
'''

lead_prompt = lead_prompt + task


user_proxy = UserProxyAgent(
    name = "supervisior",
    system_message = "A Human Head of Architecure",
    code_execution_config={
        "last_n_message":2,
        "work_dir": "groupchat",
        "use_docker": False
    },
    human_input_mode="NEVER",
)

cloud_agent = AssistantAgent(
    name = "cloud",
    system_message = cloud_prompt,
    llm_config = {"config_list": config_list}
)

oss_agent = AssistantAgent(
    name = "oss",
    system_message = oss_prompt,
    llm_config = {"config_list": config_list}
)

lead_agent = AssistantAgent(
    name = "lead",
    system_message = lead_prompt,
    llm_config = {"config_list": config_list}
)

def state_transition(last_speaker,groupchat):
    message = groupchat.messages

    if last_speaker is user_proxy:
        return cloud_agent
    elif last_speaker is cloud_agent:
        return oss_agent
    elif last_speaker is oss_agent:
        return lead_agent
    elif last_speaker is lead_agent:
        return None
    
groupchat = autogen.GroupChat(
    agents=[user_proxy,cloud_agent,oss_agent,lead_agent],
    messages=[],
    max_round=6,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager, message="Provide your best architecture based on the PAAS business requirements."
)