import json
import os
import openai
from dotenv import load_dotenv

load_dotenv('./.env')

EMBED_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def create_cluster_description(data, model="gpt-4o"):    
    system_message = '''You are an expert business analyst. Your task is to analyze groups of companies and provide a concise, meaningful description of what unifies them. Focus on identifying common themes in their:
    - Problem spaces
    - Solution approaches
    - Target users
    - Overall business positioning

    Provide your analysis in JSON format with the following structure:
    {
        "cluster_number": {
            "cluster_name": "Name of the cluster",
            "cluster_theme": "Brief 1-2 sentence theme",
            "common_characteristics": [
                "Key characteristic 1",
                "Key characteristic 2",
                "Key characteristic 3"
            ],
            "market_focus": "Primary market/industry focus"
        }
    }'''

    user_message = f'''The following text describes companies in the same cluster. Each company entry contains their problem, solution, target users, and key value proposition:

    {data}

    Analyze these companies and identify the common patterns and themes that unite them. What makes these companies similar enough to be grouped together?'''
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            response_format={'type': "json_object"}
        )

        answer = json.loads(response.choices[0].message.content.strip()) 

        return answer

    except Exception as e:
        print(f"An error occurred: {e}")



