import re
import datetime
import json
import numpy as np
import pandas as pd
import ollama
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


def get_model_response(message, dilemma):
    """Get a single response from the model."""
    try:
        stream = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'user', 'content': message},
                {'role': 'user', 'content': dilemma}
            ],
            stream=True,
        )
        
        response = ''
        for chunk in stream:
            response += chunk['message']['content']
        
        return response.strip()
    except Exception as e:
        print(f"Error getting response: {e}")
        return None


def categorize_response(response):
    """Categorize the response as A, B, or invalid."""
    if not response:
        return 'invalid'
    clean_response = response.lower().strip()
    if re.match(r'^(action\s*a|ação\s*a|aktie\s*a)[.,]?\s*$', clean_response):
        return 'A'
    elif re.match(r'^(action\s*b|ação\s*b|aktie\s*b)[.,]?\s*$', clean_response):
        return 'B'
    return 'invalid'


def run_multiple_prompts(n_runs, message, dilemma):
    """Run multiple prompts and collect responses."""
    results = []
    for i in range(n_runs):
        print(f"Running prompt {i+1}/{n_runs}")
        response = get_model_response(message, dilemma)
        category = categorize_response(response)
        results.append(category)
        print(f"Response: {response} -> Categorized as: {category}")
    
    return results


def plot_results(results):
    """Create a bar plot of the results."""
    counts = Counter(results)
    total = len(results)

    # Ensure all categories are present
    categories = ['A', 'B', 'invalid']
    percentages = {cat: (counts.get(cat, 0)/total)*100 for cat in categories}

    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(percentages.keys(), percentages.values())
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.title('Distribution of Model Responses')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)  # Set y-axis to show full percentage range
    
    # Save the plot
    plt.savefig('response_distribution.png')
    plt.close()


def calculate_binary_error(responses):
    """
    Calculate the standard error for binary responses.
    
    Args:
        responses (list): List of responses, e.g. ['A', 'B', 'A', 'A']
    
    Returns:
        tuple: (amount of 'A' responses, standard error)
    """
    n = len(responses)
    if n == 0:
        raise ValueError("Empty response list")
        
    # Calculate proportion of 'A' responses
    count_A = sum(1 for x in responses if x == 'A')
    p = count_A / n
    
    # Calculate standard error
    standard_error = np.sqrt((p * (1 - p)) / n)
    
    return count_A, standard_error


def plot_multilingual_results(filenames):
    """Create a comparative bar plot of 'A' responses across languages."""
    data = []
    
    # Load and process each file
    for filename in filenames:
        with open(filename, 'r') as f:
            exp_data = json.load(f)
            
        # Calculate percentage of 'A' responses
        results = exp_data['results']
        a_count, standard_error = calculate_binary_error(results)
        total_valid = sum(1 for r in results if r != 'invalid')
        a_percentage = (a_count / total_valid * 100) if total_valid > 0 else 0
        
        data.append({
            'language': exp_data['language'],
            'percentage_A': a_percentage,
            'standard_error': standard_error * 100
        })
        print('standard error for {} is: {}'.format(exp_data['language'], standard_error))
    
    # Create DataFrame for plotting
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df['language'], df['percentage_A'])
    plt.errorbar(df['language'], df['percentage_A'], yerr=df['standard_error'], fmt='o', color='r')
    
    """
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    """

    plt.title('Percentage of "A" Responses by Language')
    plt.ylabel('Percentage of A Responses')
    plt.ylim(0, 100)
    
    plt.savefig('multilingual_comparison.png')
    plt.close()


def load_dilemmas(filename):
    """Load dilemmas from a JSON file."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Parse each JSON object separately
    dilemmas = [json.loads(item) for item in content.strip().split('\n\n')]
    return dilemmas


def format_dilemma(dilemma, language='english'):
    """Format a dilemma dictionary into the expected string format with language-specific terms.
    
    Args:
        dilemma (dict): Dictionary containing the dilemma content
        language (str): Language code ('english', 'portuguese', or 'dutch')
    
    Returns:
        str: Formatted dilemma text
    """
    translations = {
        'english': {
            'action': 'Action'
        },
        'portuguese': {
            'action': 'Ação'
        },
        'dutch': {
            'action': 'Aktie'
        }
    }
    
    terms = translations.get(language.lower(), translations['english'])
    
    return f'''
    {dilemma['Scenario']}

    {terms['action']} A: {dilemma['Action A']}

    {terms['action']} B: {dilemma['Action B']}
    '''


def save_experiment_results(results, n_runs, language, domain, output_dir):
    """Save raw experiment results to a JSON file within the specified directory."""
    filename = f"{language}_{domain.lower().replace(' ', '_')}_results.json"
    filepath = output_dir / filename
    
    data = {
        'language': language,
        'domain': domain,
        'n_runs': n_runs,
        'results': results
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return filepath


def create_output_directory():
    """Create a timestamped directory for experiment results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiment_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def main():
    language_configs = {
        'english': {
            'file': 'english_dilemmas.txt',
            'prompt': """
            Given a scenario and two actions, which action would you recommend the agent take?
            You should only respond with either "Action A" or "Action B" without any other words.
            """
        },
        'portuguese': {
            'file': 'portuguese_dilemmas.txt',
            'prompt': """
            Dado um cenário e duas ações, que ação recomendas que o agente tome?
            Responde apenas com "Ação A" ou "Ação B", sem quaisquer outras palavras.
            """
        },
        'dutch': {
            'file': 'dutch_dilemmas.txt',
            'prompt': """
            Gegeven een scenario en twee akties, welke aktie zou je aan de agent aanbevelen?
            Geef je antwoord als "Aktie A" of "Aktie B" zonder andere woorden te gebruiken.
            """
        }
    }

    output_dir = create_output_directory()
    print(f"Created output directory: {output_dir}")

    n_runs = 1000
    for language, config in language_configs.items():
        print(f"\nProcessing {language} dilemmas...")
        
        try:
            dilemmas = load_dilemmas(config['file'])
        except FileNotFoundError:
            print(f"Warning: Dilemma file for {language} not found. Skipping...")
            continue
            
        for dilemma in dilemmas:
            print(f"Processing dilemma: {dilemma['Domain']}")
            formatted_dilemma = format_dilemma(dilemma, language=language)
            # Run experiment
            results = run_multiple_prompts(n_runs, config['prompt'], formatted_dilemma)
            
            # Save results
            save_experiment_results(
                results=results,
                n_runs=n_runs,
                language=language,
                domain=dilemma['Domain'],
                output_dir=output_dir
            )
            
    print(f"\nExperiment complete. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
