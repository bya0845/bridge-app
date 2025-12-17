"""
Simple script to test the query endpoint with the fine-tuned model.
Run the test_local_app.py server first, then run this script.
"""
import requests
import json

BASE_URL = "http://localhost:5000"

# Test queries - complex examples from training set (exact phrasing matters!)
test_queries = [
    "How many bridges are there?",
    "Find top 5 bridges by span count",
    "Top 10 bridges by spans",
    "Bridge statistics by county",
    "Orange county bridges carrying 84i with more than 4 spans",
    "Westchester bridges crossing river with at least 3 spans",
    "Bridges on 987g in Dutchess county",
    "Bridges crossing saw mill river",
    "Top 3 bridges by span count in Orange",
]

def test_query(query):
    """Test a single query."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    
    try:
        # Send query to parser endpoint
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            endpoint = data.get('endpoint')
            print(f"✓ Generated endpoint: {endpoint}")
            
            # Try calling the endpoint
            if endpoint.startswith('/api/'):
                api_url = f"{BASE_URL}{endpoint}"
                print(f"  Calling: {api_url}")
                
                api_response = requests.get(api_url, timeout=10)
                if api_response.status_code == 200:
                    api_data = api_response.json()
                    
                    if 'count' in api_data:
                        print(f"  ✓ Results: {api_data['count']} item(s)")
                    elif 'total_bridges' in api_data:
                        print(f"  ✓ Total bridges: {api_data['total_bridges']}")
                    else:
                        print(f"  ✓ Response: {json.dumps(api_data, indent=2)[:200]}...")
                else:
                    print(f"  ✗ API error: {api_response.status_code}")
        else:
            print(f"✗ Parser error: {response.status_code}")
            print(f"  {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to server")
        print("  Make sure test_local_app.py is running on port 5000")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    print("\n" + "="*70)
    print("Bridge Query Parser - Test Script")
    print("="*70)
    print(f"Testing against: {BASE_URL}")
    print(f"Total test queries: {len(test_queries)}")
    
    # Test each query
    for query in test_queries:
        test_query(query)
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

