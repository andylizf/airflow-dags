"""Utilities for fetching and processing GitHub issues."""

import os
from typing import List, Dict, Any
import requests
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class GitHubIssue:
    title: str
    body: str
    number: int
    state: str  # open/closed
    created_at: str
    comments_url: str
    html_url: str

def fetch_github_issues(
    owner: str,
    repo: str,
    token: str,
    state: str = "all"  # "open", "closed", "all"
) -> List[GitHubIssue]:
    """Fetch all issues from a GitHub repository."""
    issues = []
    page = 1
    per_page = 100
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": state,
            "page": page,
            "per_page": per_page,
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        batch = response.json()
        if not batch:
            break
            
        for issue in batch:
            # Skip pull requests
            if "pull_request" in issue:
                continue
                
            issues.append(GitHubIssue(
                title=issue["title"],
                body=issue["body"] or "",
                number=issue["number"],
                state=issue["state"],
                created_at=issue["created_at"],
                comments_url=issue["comments_url"],
                html_url=issue["html_url"]
            ))
        
        page += 1
    
    return issues

def fetch_issue_comments(comments_url: str, token: str) -> List[Dict[str, Any]]:
    """Fetch all comments for an issue."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(comments_url, headers=headers)
    response.raise_for_status()
    
    return response.json()

def save_issues_to_dataset(
    issues: List[GitHubIssue],
    output_dir: Path,
    include_comments: bool = True,
    token: str = None
):
    """Save issues to dataset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for issue in issues:
        issue_dir = output_dir / str(issue.number)
        issue_dir.mkdir(exist_ok=True)
        
        # Save main issue content
        with open(issue_dir / "issue.json", "w") as f:
            json.dump(issue.__dict__, f, indent=2)
        
        # Save comments if requested
        if include_comments and token:
            comments = fetch_issue_comments(issue.comments_url, token)
            if comments:
                with open(issue_dir / "comments.json", "w") as f:
                    json.dump(comments, f, indent=2)

def filter_issues_with_llm(
    model,
    tokenizer,
    issues: List[GitHubIssue],
    batch_size: int = 8
) -> List[GitHubIssue]:
    """Filter issues to keep only real questions/problems using LLM."""
    filtered_issues = []
    
    prompt_template = """
    Analyze if the following GitHub issue is a real technical question or problem that needs solving.
    Respond with only 'YES' if it's a real question/problem, or 'NO' if it's not.
    
    Issue Title: {title}
    Issue Body: {body}
    """
    
    for i in range(0, len(issues), batch_size):
        batch = issues[i:i + batch_size]
        prompts = [
            prompt_template.format(
                title=issue.title,
                body=issue.body[:1000]  # Truncate long bodies
            )
            for issue in batch
        ]
        
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048
        )
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            num_return_sequences=1,
        )
        
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for issue, response in zip(batch, responses):
            if "YES" in response.upper():
                filtered_issues.append(issue)
    
    return filtered_issues

def generate_issue_responses(
    model,
    tokenizer,
    issues: List[GitHubIssue],
    batch_size: int = 4
) -> Dict[int, str]:
    """Generate responses for filtered issues using fine-tuned model."""
    responses = {}
    
    prompt_template = """
    Given the following GitHub issue, provide a helpful and technical response:
    
    Issue Title: {title}
    Issue Body: {body}
    
    Response:
    """
    
    for i in range(0, len(issues), batch_size):
        batch = issues[i:i + batch_size]
        prompts = [
            prompt_template.format(
                title=issue.title,
                body=issue.body[:2000]  # Truncate long bodies
            )
            for issue in batch
        ]
        
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048
        )
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=1,
            temperature=0.7,
        )
        
        responses_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for issue, response in zip(batch, responses_text):
            responses[issue.number] = response.split("Response:")[-1].strip()
    
    return responses 