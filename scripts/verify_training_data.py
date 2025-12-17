#!/usr/bin/env python3
"""
Verify training_data.json endpoints are correct and properly formatted
"""
import json
from pathlib import Path
from collections import Counter
import re

def verify_training_data(json_path):
    """Verify training data endpoints"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Training Data Verification")
    print(f"{'='*70}\n")
    print(f"Total examples: {len(data)}")
    
    # Collect all outputs (endpoints)
    endpoints = [item['output'] for item in data]
    endpoint_counter = Counter(endpoints)
    
    # Group endpoints by pattern
    endpoint_patterns = {
        'count': [],
        'list': [],
        'search': [],
        'search_with_params': [],
        'group_by': [],
        'inspections': [],
        'other': []
    }
    
    for endpoint in set(endpoints):
        if endpoint == '/api/bridges/count':
            endpoint_patterns['count'].append(endpoint)
        elif endpoint == '/api/bridges':
            endpoint_patterns['list'].append(endpoint)
        elif endpoint.startswith('/api/bridges/search?'):
            # Parse parameters
            params = endpoint.split('?')[1] if '?' in endpoint else ''
            endpoint_patterns['search_with_params'].append(endpoint)
        elif endpoint.startswith('/api/bridges/group-by'):
            endpoint_patterns['group_by'].append(endpoint)
        elif endpoint.startswith('/api/inspections'):
            endpoint_patterns['inspections'].append(endpoint)
        else:
            endpoint_patterns['other'].append(endpoint)
    
    # Print summary by category
    print(f"\n{'='*70}")
    print(f"Endpoint Categories")
    print(f"{'='*70}")
    
    for category, eps in endpoint_patterns.items():
        if eps:
            print(f"\n{category.upper()}: {len(eps)} unique endpoint(s)")
            for ep in sorted(eps)[:5]:  # Show first 5
                count = endpoint_counter[ep]
                print(f"  {count:4d}x {ep}")
            if len(eps) > 5:
                print(f"  ... and {len(eps) - 5} more")
    
    # Analyze search parameters
    print(f"\n{'='*70}")
    print(f"Search Parameter Analysis")
    print(f"{'='*70}")
    
    params_used = {
        'county': 0,
        'carried': 0,
        'crossed': 0,
        'spans': 0,
        'min_spans': 0,
        'max_spans': 0,
        'sort': 0,
        'order': 0,
        'limit': 0
    }
    
    param_combinations = Counter()
    
    for endpoint in endpoint_patterns['search_with_params']:
        if '?' in endpoint:
            params_str = endpoint.split('?')[1]
            params = params_str.split('&')
            param_names = [p.split('=')[0] for p in params]
            
            for param in param_names:
                if param in params_used:
                    params_used[param] += 1
            
            param_combinations[tuple(sorted(param_names))] += 1
    
    print("\nParameter Usage:")
    for param, count in sorted(params_used.items(), key=lambda x: -x[1]):
        print(f"  {param:12s}: {count:4d} times")
    
    print(f"\nTop Parameter Combinations:")
    for combo, count in param_combinations.most_common(10):
        print(f"  {count:4d}x {', '.join(combo)}")
    
    # Check for issues
    print(f"\n{'='*70}")
    print(f"Validation Checks")
    print(f"{'='*70}")
    
    issues = []
    
    # Check for malformed URLs
    for i, item in enumerate(data):
        endpoint = item['output']
        
        # Check for basic format
        if not endpoint.startswith('/api/'):
            issues.append(f"Line {i+1}: Endpoint doesn't start with /api/: {endpoint}")
        
        # Check for double separators
        if '&&' in endpoint or '??' in endpoint:
            issues.append(f"Line {i+1}: Double separator in endpoint: {endpoint}")
        
        # Check for spaces in URL
        if ' ' in endpoint:
            issues.append(f"Line {i+1}: Space in endpoint: {endpoint}")
        
        # Check for missing values
        if '=' in endpoint:
            params = endpoint.split('?')[1].split('&') if '?' in endpoint else []
            for param in params:
                if '=' in param:
                    key, val = param.split('=', 1)
                    if not val:
                        issues.append(f"Line {i+1}: Empty value for {key}: {endpoint}")
    
    if issues:
        print(f"\n[WARNING] Found {len(issues)} issue(s):")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print(f"\n[OK] No issues found! All endpoints are properly formatted.")
    
    # Sample examples
    print(f"\n{'='*70}")
    print(f"Sample Examples")
    print(f"{'='*70}")
    
    sample_categories = [
        ('County search', lambda x: 'county=' in x['output']),
        ('Span filter', lambda x: 'spans=' in x['output'] or 'min_spans=' in x['output']),
        ('Multi-parameter', lambda x: x['output'].count('&') >= 2),
        ('Sorting', lambda x: 'sort=' in x['output']),
        ('Inspections', lambda x: '/inspections' in x['output']),
    ]
    
    for category_name, condition in sample_categories:
        examples = [item for item in data if condition(item)]
        if examples:
            print(f"\n{category_name}: ({len(examples)} examples)")
            sample = examples[0]
            print(f"  Input:  {sample['input']}")
            print(f"  Output: {sample['output']}")
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Verification Complete!")
    print(f"{'='*70}\n")


def verify_training_safety(json_path):
    """Check for potentially unsafe endpoints in training data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    UNSAFE_PATTERNS = [
        'create',
        'delete',
        'upload',
        'update',
        'modify',
        'remove',
    ]
    
    print(f"\n{'='*70}")
    print(f"Safety Verification - Checking for unsafe endpoints")
    print(f"{'='*70}")
    print(f"\nChecking {len(data)} training examples...")
    
    unsafe_examples = []
    for item in data:
        endpoint = item['output'].lower()
        
        for pattern in UNSAFE_PATTERNS:
            if pattern in endpoint:
                unsafe_examples.append({
                    'input': item['input'],
                    'output': item['output'],
                    'pattern': pattern
                })
                break
    
    if unsafe_examples:
        print(f"\n[WARNING] Found {len(unsafe_examples)} potentially unsafe endpoint(s):\n")
        for ex in unsafe_examples:
            print(f"  Pattern: {ex['pattern']}")
            print(f"  Input:   {ex['input']}")
            print(f"  Output:  {ex['output']}")
            print()
        print(f"[ACTION REQUIRED] Remove these from training data!")
        print(f"The model should ONLY learn read-only operations (GET requests).")
    else:
        print(f"\n[OK] All {len(data)} endpoints are safe (read-only)")
        print(f"No create/delete/upload/update/modify/remove patterns found.")
    
    print(f"\n{'='*70}\n")
    return len(unsafe_examples) == 0


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    json_path = script_dir.parent / "src" / "data" / "training_data.json"
    
    if not json_path.exists():
        print(f"[ERROR] Training data not found at {json_path}")
        exit(1)
    
    verify_training_data(json_path)
    is_safe = verify_training_safety(json_path)
    
    if not is_safe:
        exit(1)

