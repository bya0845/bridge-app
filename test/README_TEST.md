# Local Testing Guide

This directory contains files for local testing using SQLite instead of MySQL.

## Setup

1. **Make sure you've trained the model:**
   ```bash
   cd ..
   python parser\train_model.py
   ```

2. **Database will be created automatically** at `test/test_bridges.db` on first run

## Running the Test Server

From the project root:
```bash
cd test
python test_local_app.py
```

Or from the test directory:
```bash
python test_local_app.py
```

The server will:
- Create `test_bridges.db` if it doesn't exist
- Import data from `../data/bridge_data.csv` 
- Load the fine-tuned model from `../fine_tuned_model/`
- Start on http://localhost:5000

## Testing the Query Endpoint

### Option 1: Use the test script
```bash
# In a new terminal
cd test
python test_queries.py
```

### Option 2: Manual testing with curl
```bash
# Test query parsing
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Show me bridges in Westchester\"}"

# Expected response:
# {"query": "Show me bridges in Westchester", "endpoint": "/api/bridges/search?county=Westchester"}
```

### Option 3: Test in Python
```python
import requests

response = requests.post(
    'http://localhost:5000/query',
    json={'query': 'How many bridges are there?'}
)
print(response.json())
```

## Available Endpoints

### Natural Language Query
- **POST** `/query` - Parse natural language to API endpoint
  ```json
  {"query": "Show me bridges in Westchester"}
  ```

### Bridge Endpoints
- **GET** `/api/bridges/count` - Get total bridge count
- **GET** `/api/bridges` - List all bridges (paginated)
- **GET** `/api/bridges/search?county=X&min_spans=Y` - Search bridges

### Inspection Endpoints
- **POST** `/api/inspections/create` - Create inspection
- **GET** `/api/inspections/<bin>` - Get inspections for a bridge
- **GET** `/api/inspections` - Get all inspections

## Test Queries Examples

```bash
# Count query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"How many bridges are there?\"}"

# County search
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Bridges in Orange county\"}"

# Span filter
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Show bridges with 5+ spans\"}"

# Top N query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Top 10 bridges by span count\"}"
```

## Troubleshooting

### Model not found
```
Error: Model path '../fine_tuned_model' does not exist
```
**Solution:** Train the model first:
```bash
cd ..
python parser\train_model.py
```

### Database errors
```
Error: no such table: bridges
```
**Solution:** Delete `test_bridges.db` and restart - it will be recreated

### Import errors
```
ModuleNotFoundError: No module named 'config'
```
**Solution:** Make sure you're running from the correct directory with proper Python path

## Files

- `test_local_app.py` - Flask app using SQLite
- `test_queries.py` - Automated test script
- `test_bridges.db` - SQLite database (created automatically)
- `test_params.py` - Parameter testing utilities

