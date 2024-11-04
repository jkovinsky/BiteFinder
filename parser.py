import os
import pandas as pd
import random
import re
import time
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

## WARNING: Look out for ERROR RESPONSE 429 (REQUEST LIMIT EXCEEDED) ##



# set up parser

load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')
llm = "mistral-large-latest"

class Parser:
    def __init__(self, menu_content):
        self.menu_content = menu_content
    
    def decideBool(self):
        menu_template_bool = """
        Does the price come before each food item?
        Answer True if yes, False if not. 

        Strictly format the output as either True or False.
        If prices are not found, output -1. 

        text: {text}
        """

        prompt_bool = ChatPromptTemplate.from_template(template=menu_template_bool)

        message_bool = prompt_bool.format_messages(text=self.menu_content)
        chat_bool = ChatMistralAI(temperature=0.0, model=llm, api_key=api_key)
        response_bool = chat_bool.invoke(message_bool).content
        response_bool = response_bool.lower().capitalize() # ensure format is True/False

        return response_bool
    
    def parse(self, decision):
        menu_template_A = """
        Given that the price comes before each food item, \
        extract the price and its corresponding item as a vector.

        Strictly format the output as a comma separated list \
        where each element is a vector of <price, item>

        text: {text}
        """

        menu_template_B = """
        Given that the item comes before each price, \
        extract the item and its corresponding price as a vector.

        Strictly format the output as a comma separated list \
        where each element is a vector of <price, item>

        text: {text}
        """

        if decision == 'True' or decision == 'False':
                
            if decision == 'True':
                prompt_template = ChatPromptTemplate.from_template(template=menu_template_A)
            else:
                prompt_template = ChatPromptTemplate.from_template(template=menu_template_B)
            
            message  = prompt_template.format_messages(text=self.menu_content)
            chat     = ChatMistralAI(temperature=0.0, model=llm, api_key=api_key)
            response = chat.invoke(message)
            output   = response.content
            
            pattern = r"\<(.*?)\>"
            matches = re.findall(pattern, output)
            result = [None] * len(matches)

            for idx, match in enumerate(matches):
                entry = match.split(", ", 1)

                if "$" in entry[0]:
                    entry[0] = entry[0].replace("$", "")
                try: 
                    entry[0] = float(entry[0])
                except ValueError:
                    entry[0] = None
                
                entry[1] = entry[1].strip(" ")
                result[idx] = entry

            return result
            
        elif decision == '-1':
            print('Prices were not discoverable.')
            result = None
            return result

        else:
            print("Invalid model response: did not return True/False.")
            result = None
            return result 


if __name__ == "__main__":
    ### TEST ####
    data = pd.read_csv('allMenus.csv')
    menus = data[data['menu'] == 'Y']
    random_idx = random.choice(menus.index)
    random_menu_content = menus['content'][random_idx]
    random_link = menus['link'][random_idx]
    print(random_menu_content)
    len_menu = len(random_menu_content)
    
    parser = Parser(random_menu_content)
    decision = parser.decideBool()
    print(decision)
    result = parser.parse(decision)

    with open("llmTest/testParser.txt", "a") as file:
        file.write(f"link: {random_link}\n")
        file.write(f"Price/Item: {result}\n")
        file.write(f"Size of request: {len_menu}\n")
        file.write(f"Size of menu: {len(result)}\n")
        file.write("===============================\n")



