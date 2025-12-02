import requests
import re

def parse_github_url(url):
    """Extracts owner and repo name from a GitHub URL."""
    # Regex to capture owner/repo from various URL formats
    match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if match:
        return match.group(1), match.group(2).removesuffix('.git')
    return None, None

def fetch_recent_commits(repo_url, limit=10):
    """
    Fetches the most recent commits from a public GitHub repository.
    Returns a list of dictionaries with 'sha', 'message', and 'author'.
    """
    owner, repo = parse_github_url(repo_url)
    if not owner or not repo:
        return {"error": "Invalid GitHub URL"}

    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page={limit}"
    
    try:
        response = requests.get(api_url)
        
        if response.status_code == 200:
            commits_data = response.json()
            cleaned_commits = []
            for item in commits_data:
                cleaned_commits.append({
                    "sha": item['sha'][:7], # Short SHA
                    "message": item['commit']['message'],
                    "author": item['commit']['author']['name'],
                    "date": item['commit']['author']['date']
                })
            return {"success": True, "data": cleaned_commits}
        elif response.status_code == 404:
            return {"error": "Repository not found or private."}
        elif response.status_code == 403:
            return {"error": "API Rate limit exceeded. Try again later."}
        else:
            return {"error": f"GitHub API Error: {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}
