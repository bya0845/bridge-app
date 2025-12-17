from flask import Flask, jsonify, request, render_template
import sqlite3
import os
import sys
import json
import pandas as pd
import requests
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser.inference import BridgeQueryParser
from config.logging_config import setup_logging, get_logger

# Configure logging
setup_logging()
logger = get_logger(__name__)

app = Flask(__name__,
           template_folder='../templates',
           static_folder='../static')

COUNTY_MAP = {
    '1': 'Columbia',
    '2': 'Dutchess',
    '3': 'Orange',
    '4': 'Putnam',
    '5': 'Rockland',
    '6': 'Ulster',
    '7': 'Westchester'
}

# Initialize the query parser with fine-tuned model
logger.info("Loading BridgeQueryParser with fine-tuned model...")
try:
    parser = BridgeQueryParser(model_path="../fine_tuned_model")
    logger.info("BridgeQueryParser loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    parser = None

def get_db_connection():
    conn = sqlite3.connect('test_bridges.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    """Create SQLite tables and import CSV data if needed."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create bridges table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bridges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bin VARCHAR(50) UNIQUE,
        carried VARCHAR(100),
        crossed VARCHAR(100),
        county VARCHAR(100),
        spans INTEGER,
        latitude DECIMAL(10, 6),
        longitude DECIMAL(10, 6),
        google_map_link VARCHAR(500)
    )
    """)

    # Create inspections table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bridge_id INTEGER NOT NULL,
        bin VARCHAR(50) NOT NULL,
        inspection_date DATE NOT NULL,
        inspection_time TIME NOT NULL,
        weather VARCHAR(50),
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (bridge_id) REFERENCES bridges(id)
    )
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM bridges")
    count = cursor.fetchone()[0]

    if count == 0 and os.path.exists('../data/bridge_data.csv'):
        print("📥 Importing bridge data from CSV...")
        df = pd.read_csv('../data/bridge_data.csv')

        for index, row in df.iterrows():
            cursor.execute("""
            INSERT INTO bridges (bin, carried, crossed, county, spans, latitude, longitude, google_map_link)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['bin'],
                row['carried'],
                row['crossed'],
                row['county'],
                row['spans'],
                row['latitude'],
                row['longitude'],
                row['google map link']
            ))

        conn.commit()
        print(f"✓ Imported {len(df)} bridges\n")

    cursor.close()
    conn.close()

create_tables()

def map_county(county_num):
    return COUNTY_MAP.get(str(county_num), str(county_num))

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def parse_query():
    """Natural language query endpoint using fine-tuned model."""
    try:
        data = request.get_json()
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        if not parser:
            return jsonify({"error": "Model not loaded"}), 503

        logger.info(f"Parsing query: {user_query}")
        
        # Use the fine-tuned model to translate query to API endpoint
        endpoint = parser.parse_query(user_query)
        
        if not endpoint:
            return jsonify({"error": "Failed to parse query"}), 400

        logger.info(f"Generated endpoint: {endpoint}")
        
        return jsonify({
            "query": user_query,
            "endpoint": endpoint
        })

    except Exception as e:
        logger.error(f"Error in parse_query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent/status')
def agent_status():
    """Check if the AI agent is ready."""
    if parser:
        return jsonify({"status": "ready", "message": "Agent is ready"})
    else:
        return jsonify({"status": "error", "error": "Model not loaded"}), 503

@app.route('/api/agent/query', methods=['POST'])
def agent_query():
    """AI Agent query endpoint - uses fine-tuned model to answer questions."""
    try:
        data = request.get_json()
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({"success": False, "error": "Please provide a question about bridge data."}), 400

        if not parser:
            return jsonify({"success": False, "error": "The AI assistant is currently unavailable. Please try again later."}), 503

        logger.info(f"Agent query: {user_query}")
        
        # Parse the query to get the API endpoint
        endpoint = parser.parse_query(user_query)
        
        if not endpoint:
            return jsonify({"success": False, "error": "I couldn't understand your question. Please try rephrasing it."}), 400

        logger.info(f"Generated endpoint: {endpoint}")
        
        # Call the API endpoint and get results
        try:
            # Make internal request to our own API
            base_url = request.url_root.rstrip('/')
            api_url = f"{base_url}{endpoint}"
            
            logger.info(f"Calling API: {api_url}")
            api_response = requests.get(api_url, timeout=10)
            
            if api_response.status_code == 200:
                api_data = api_response.json()
                
                # Format the response for display
                if 'count' in api_data and 'results' in api_data:
                    count = api_data['count']
                    results = api_data['results']
                    
                    if count == 0:
                        result_text = f"**Query:** {user_query}\n\n**API Endpoint:** `{endpoint}`\n\n**Result:** No bridges found matching your criteria."
                    else:
                        # Create table with ALL results
                        result_text = f"**Query:** {user_query}\n\n**API Endpoint:** `{endpoint}`\n\n**Found {count} bridge(s)**\n\n"
                        
                        # Build markdown table
                        result_text += "| # | BIN | County | Carries | Crosses | Spans |\n"
                        result_text += "|---|-----|--------|---------|---------|-------|\n"
                        
                        # Show ALL results in table format
                        for i, bridge in enumerate(results, 1):
                            bin_val = str(bridge.get('bin', 'N/A'))
                            county = str(bridge.get('county', 'N/A'))
                            carried = str(bridge.get('carried', 'N/A'))[:40]  # Truncate long values
                            crossed = str(bridge.get('crossed', 'N/A'))[:40]
                            spans = str(bridge.get('spans', 'N/A'))
                            
                            result_text += f"| {i} | {bin_val} | {county} | {carried} | {crossed} | {spans} |\n"
                        
                        result_text += f"\n**Total Results:** {count} bridge(s)"
                
                elif 'total_bridges' in api_data:
                    total = api_data['total_bridges']
                    result_text = f"**Query:** {user_query}\n\n**API Endpoint:** `{endpoint}`\n\n**Total Bridges in Database:** {total}"
                
                elif 'inspections' in api_data:
                    # Handle inspection results
                    count = api_data.get('count', 0)
                    inspections = api_data['inspections']
                    
                    if count == 0:
                        result_text = f"**Query:** {user_query}\n\n**API Endpoint:** `{endpoint}`\n\n**Result:** No inspections found."
                    else:
                        result_text = f"**Query:** {user_query}\n\n**API Endpoint:** `{endpoint}`\n\n**Found {count} inspection(s)**\n\n"
                        
                        # Build inspection table
                        result_text += "| # | BIN | Date | Time | Weather | Notes |\n"
                        result_text += "|---|-----|------|------|---------|-------|\n"
                        
                        for i, insp in enumerate(inspections, 1):
                            bin_val = str(insp.get('bin', 'N/A'))
                            date = str(insp.get('inspection_date', 'N/A'))
                            time = str(insp.get('inspection_time', 'N/A'))
                            weather = str(insp.get('weather', 'N/A'))
                            notes = str(insp.get('notes', 'N/A'))[:50]  # Truncate long notes
                            
                            result_text += f"| {i} | {bin_val} | {date} | {time} | {weather} | {notes} |\n"
                        
                        result_text += f"\n**Total Results:** {count} inspection(s)"
                
                elif isinstance(api_data, list):
                    # Handle list responses (like group-by queries)
                    result_text = f"**Query:** {user_query}\n\n**API Endpoint:** `{endpoint}`\n\n**Results:**\n\n"
                    
                    if len(api_data) > 0:
                        # Get headers from first item
                        headers = list(api_data[0].keys())
                        
                        # Build table header
                        result_text += "| " + " | ".join(headers) + " |\n"
                        result_text += "|" + "|".join(["---"] * len(headers)) + "|\n"
                        
                        # Build table rows
                        for row in api_data:
                            values = [str(row.get(h, 'N/A')) for h in headers]
                            result_text += "| " + " | ".join(values) + " |\n"
                        
                        result_text += f"\n**Total Results:** {len(api_data)} row(s)"
                    else:
                        result_text += "No data found."
                
                else:
                    # Generic JSON response
                    result_text = f"**Query:** {user_query}\n\n**API Endpoint:** `{endpoint}`\n\n**Result:**\n```json\n{json.dumps(api_data, indent=2)}\n```"
                
                return jsonify({"success": True, "result": result_text})
            elif api_response.status_code == 404:
                return jsonify({
                    "success": False, 
                    "error": "I couldn't comprehend your query. Please try rephrasing it or ask about bridge data."
                })
            else:
                return jsonify({
                    "success": False, 
                    "error": f"I encountered an issue processing your request. Please try again or rephrase your question."
                })
                
        except Exception as api_error:
            logger.error(f"Error calling API: {api_error}")
            return jsonify({
                "success": False,
                "error": f"I understood your query but couldn't fetch the results. Please try again in a moment."
            })

    except Exception as e:
        logger.error(f"Error in agent_query: {e}")
        return jsonify({"success": False, "error": "I encountered an unexpected issue. Please try rephrasing your question."}), 500

@app.route('/api/bridges/count')
def bridges_count():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM bridges")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return jsonify({"total_bridges": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/bridges')
def list_bridges():
    """List all bridges with optional pagination."""
    try:
        limit = request.args.get('limit', type=int, default=100)
        offset = request.args.get('offset', type=int, default=0)

        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM bridges LIMIT ? OFFSET ?", (limit, offset))
        bridges = [dict(row) for row in cursor.fetchall()]

        for bridge in bridges:
            bridge['county'] = map_county(bridge['county'])

        cursor.execute("SELECT COUNT(*) FROM bridges")
        total = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return jsonify({"count": len(bridges), "total": total, "results": bridges})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/bridges/search')
def search_bridges():
    try:
        bin_id = request.args.get('bin')
        county = request.args.get('county')
        carried = request.args.get('carried')
        crossed = request.args.get('crossed')
        min_spans = request.args.get('min_spans', type=int)
        max_spans = request.args.get('max_spans', type=int)
        sort = request.args.get('sort', 'id')
        order = request.args.get('order', 'asc')
        limit = request.args.get('limit', type=int)

        query = "SELECT * FROM bridges WHERE 1=1"
        params = []

        if bin_id:
            query += " AND bin LIKE ?"
            params.append(f"%{bin_id}%")

        if county:
            county_num = None
            for num, name in COUNTY_MAP.items():
                if name.lower() == county.lower():
                    county_num = num
                    break
            if county_num:
                query += " AND county = ?"
                params.append(county_num)

        if carried:
            query += " AND carried LIKE ?"
            params.append(f"%{carried}%")

        if crossed:
            query += " AND crossed LIKE ?"
            params.append(f"%{crossed}%")

        if min_spans:
            query += " AND spans >= ?"
            params.append(min_spans)

        if max_spans:
            query += " AND spans <= ?"
            params.append(max_spans)

        # Add sorting
        valid_sort_fields = ['id', 'bin', 'spans', 'carried', 'crossed', 'county']
        if sort in valid_sort_fields:
            query += f" ORDER BY {sort}"
            if order.lower() == 'desc':
                query += " DESC"
            else:
                query += " ASC"

        # Add limit
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        bridges = [dict(row) for row in cursor.fetchall()]

        for bridge in bridges:
            bridge['county'] = map_county(bridge['county'])

        cursor.close()
        conn.close()

        return jsonify({"count": len(bridges), "results": bridges})
    except Exception as e:
        logger.error(f"Error in search_bridges: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/bridges/group-by-county')
def group_by_county():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT county, COUNT(*) as count, AVG(spans) as avg_spans
            FROM bridges
            GROUP BY county
            ORDER BY county
        """)
        results = [dict(row) for row in cursor.fetchall()]

        for result in results:
            result['county'] = map_county(result['county'])

        cursor.close()
        conn.close()

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/bridges/group-by-spans')
def group_by_spans():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT spans, COUNT(*) as count
            FROM bridges
            GROUP BY spans
            ORDER BY spans
        """)
        results = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Inspection endpoints
@app.route('/api/inspections/create', methods=['POST'])
def create_inspection():
    try:
        data = request.json
        bin_id = data.get('bin')
        inspection_date = data.get('inspection_date')
        inspection_time = data.get('inspection_time')
        weather = data.get('weather')
        notes = data.get('notes')

        if not bin_id or not inspection_date or not inspection_time:
            return jsonify({"error": "Missing required fields"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get bridge_id from bin
        cursor.execute("SELECT id FROM bridges WHERE bin = ?", (bin_id,))
        result = cursor.fetchone()

        if not result:
            cursor.close()
            conn.close()
            return jsonify({"error": "Bridge not found"}), 404

        bridge_id = result[0]

        cursor.execute("""
            INSERT INTO inspections (bridge_id, bin, inspection_date, inspection_time, weather, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (bridge_id, bin_id, inspection_date, inspection_time, weather, notes))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"success": True, "message": "Inspection logged successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/inspections/<bin_id>')
def get_inspections(bin_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, bin, inspection_date, inspection_time, weather, notes, created_at
            FROM inspections
            WHERE bin = ?
            ORDER BY inspection_date DESC, inspection_time DESC
        """, (bin_id,))

        inspections = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return jsonify({"count": len(inspections), "inspections": inspections})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/inspections')
def get_all_inspections():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, bin, inspection_date, inspection_time, weather, notes, created_at
            FROM inspections
            ORDER BY inspection_date DESC, inspection_time DESC
        """)

        inspections = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return jsonify({"count": len(inspections), "inspections": inspections})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/inspections/delete-by-bridge/<bin_id>', methods=['DELETE'])
def delete_inspections_by_bridge(bin_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM inspections WHERE bin = ?", (bin_id,))
        deleted_count = cursor.rowcount

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"success": True, "message": f"Deleted {deleted_count} inspection(s) for bridge {bin_id}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/inspections/delete-all', methods=['DELETE'])
def delete_all_inspections():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM inspections")
        deleted_count = cursor.rowcount

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"success": True, "message": f"Deleted all {deleted_count} inspection(s)"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("Bridge Inspection App - LOCAL TEST MODE")
    logger.info("Running on http://localhost:5000")
    logger.info("Using SQLite database: test_bridges.db")
    logger.info("Auto-reload enabled - edit files to see changes")
    logger.info("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
