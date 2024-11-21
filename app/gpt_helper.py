import json
import os
import openai
from dotenv import load_dotenv

load_dotenv('./.env')

EMBED_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def create_cluster_description(data, model="gpt-4o"):    
    system_message = '''

'''
    
    user_message = f''''''

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8,
            response_format={'type': "json_object"}
        )

        answer = json.loads(response.choices[0].message.content.strip()) 

        return answer

    except Exception as e:
        print(f"An error occurred: {e}")



