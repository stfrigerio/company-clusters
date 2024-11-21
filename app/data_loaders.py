import json
import pandas as pd
from typing import List, Dict

def load_json_data(file: str) -> List[Dict[str, str]]:
    """
    Load and parse company data from a JSON file, extracting Summary problem, solution, users and punchline.
    
    Args:
        file (str): Path to the JSON file
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing extracted company information
    """

    with open(file) as json_file:
        data = json.load(json_file)

    extracted_data = []
    for company, company_data in data.items():
        if 'total_usage' not in company_data:
            continue

        if 'id' not in company_data:
            continue

        iteration = company_data['total_usage']['total_iterations']

        # seed_solution = company_data['solution']['seed_solution']['text']

        summary_problem = company_data['problem']['summary_problem']['text']
        summary_solution = company_data['solution'][f'summary_solution_{iteration}']['text']
        summary_target_users = company_data['users']['summary_users']['text']

        punchline = company_data['punchline']['text']

        url = company_data['total_usage']['urls_scraped'][0]

        id = company_data['id']

        extracted_data.append({
            'Company': company,
            'id': id,
            'URL': url,
            # 'Seed Solution': seed_solution,
            'Summary Problem': summary_problem,
            'Summary Solution': summary_solution,
            'Summary Users': summary_target_users,
            'Punchline': punchline
        })

    return extracted_data

def create_df(data: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame the data and combines the text information to simplify the semantic embeddings.
    
    Args:
        data (List[Dict[str, str]]): List of dictionaries containing company information
        
    Returns:
        pd.DataFrame: DataFrame with company data and additional combined_text column
    """

    df = pd.DataFrame(data)

    df['combined_text'] = df['Summary Problem'] + " " + df['Summary Solution'] + " " + df['Summary Users'] + " " + df['Punchline']
    
    return df