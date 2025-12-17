#!./venv/bin/python3.13
import sys
import csv
import os
import threading
import requests
import uuid
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    Response,
    send_from_directory,
)
from src.parser.inference import BridgeQueryParser
from dotenv import load_dotenv
from config.logging_config import configure_logger

configure_logger(log_level="INFO")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="src/templates",
    static_folder="src/static",
)

load_dotenv()

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "results/checkpoints/t5-small_best.pth")

logger.info(f"Loading BridgeQueryParser model from {MODEL_CHECKPOINT_PATH}...")
parser = BridgeQueryParser(model_path=MODEL_CHECKPOINT_PATH)
logger.info("BridgeQueryParser loaded successfully")
_parser_lock = threading.Lock()

# ============================================================================
# CONFIGURATION
# ============================================================================

COUNTY_MAP = {
    "1": "Columbia",
    "2": "Dutchess",
    "3": "Orange",
    "4": "Putnam",
    "5": "Rockland",
    "6": "Ulster",
    "7": "Westchester",
}

WEATHER_OPTIONS = [
    "Clear",
    "Cloudy",
    "Rain",
    "Light Rain",
    "Light Snow",
    "Mostly Cloudy",
]

# AI-generated endpoints whitelist (read-only operations)
ALLOWED_READ_ENDPOINTS = {
    "/api/bridges/count",
    "/api/bridges",
    "/api/bridges/search",
    "/api/bridges/by-carried",
    "/api/bridges/by-crossed",
    "/api/bridges/by-county",
    "/api/bridges/by-spans",
    "/api/bridges/group-by-county",
    "/api/bridges/group-by-spans",
    "/api/inspections",
}

LOCAL_UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
logger.info(f"Using local file storage: {LOCAL_UPLOAD_DIR}")
Path(LOCAL_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

base_url = os.getenv("FLASK_BASE_URL")

# ============================================================================
# DATABASE UTILITIES
# ============================================================================


@contextmanager
def get_db():
    """
    Context manager for database connections.
    Ensures connections are always properly closed.
    """
    db_path = os.getenv("SQLITE_PATH", "./src/data/bridges.db")

    if not os.path.exists(db_path):
        logger.error(f"SQLite database not found: {db_path}")
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def map_county(county_num):
    """Convert county number to county name."""
    return COUNTY_MAP.get(str(county_num), str(county_num))


def dict_row_to_bridge(row):
    """Convert database row to bridge dictionary with mapped county."""
    bridge = dict(row)
    bridge["county"] = map_county(bridge["county"])
    return bridge


def execute_bridge_query(query, params):
    """Execute query and return bridges with mapped counties."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict_row_to_bridge(row) for row in rows]


# ============================================================================
# AI QUERY SECURITY
# ============================================================================


def validate_endpoint(endpoint):
    """
    Ensure AI-generated endpoint is read-only.

    Returns:
        bool: True if endpoint is safe (read-only), False otherwise
    """
    # Remove query parameters
    base_endpoint = endpoint.split("?")[0]

    # Check if it starts with any allowed endpoint
    for allowed in ALLOWED_READ_ENDPOINTS:
        if base_endpoint.startswith(allowed):
            return True

    # Check for specific patterns like /api/bridges/by-county/Westchester
    if base_endpoint.startswith("/api/bridges/by-"):
        return True

    # /api/inspections/<bin_id> is safe (GET only)
    if base_endpoint.startswith("/api/inspections/"):
        parts = base_endpoint.split("/")
        if len(parts) == 4 and parts[3]:  # Has bin_id
            return True

    logger.warning(f"Endpoint validation failed: {endpoint}")
    return False


# ============================================================================
# AI QUERY ENDPOINT
# ============================================================================


@app.route("/query", methods=["POST"])
def parse_query():
    """Parse natural language query to API endpoint using AI model."""
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        with _parser_lock:
            endpoint = parser.parse_query(user_query)

        # SECURITY: Validate endpoint is read-only
        if not validate_endpoint(endpoint):
            logger.warning(f"Rejected unsafe AI-generated endpoint: '{user_query}' → {endpoint}")
            return (
                jsonify(
                    {
                        "error": "Generated endpoint not allowed (read-only queries only)",
                        "endpoint": endpoint,
                        "status": "rejected",
                    }
                ),
                403,
            )

        return jsonify({"query": user_query, "endpoint": endpoint, "status": "success"})

    except Exception as e:
        logger.error(f"Query parsing error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": "loaded"})


@app.route("/api/agent/status", methods=["GET"])
def agent_status():
    """Check if the AI agent is ready."""
    return jsonify({"status": "ready", "model": "t5-small", "message": "AI Assistant is online"})


@app.route("/api/agent/query", methods=["POST"])
def agent_query():
    """Process a natural language query and return formatted results."""
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"success": False, "error": "Please enter a question"}), 400

    try:
        # Parse the query to get API endpoint
        with _parser_lock:
            endpoint = parser.parse_query(user_query)

        # SECURITY: Validate endpoint is read-only
        if not validate_endpoint(endpoint):
            logger.warning(f"Rejected unsafe AI-generated endpoint: '{user_query}' → {endpoint}")
            return (
                jsonify({"success": False, "error": "The agent cannot comprehend your query", "endpoint": endpoint}),
                400,
            )

        # Call the generated API endpoint
        full_url = f"http://127.0.0.1:{request.environ.get('SERVER_PORT', 5001)}{endpoint}"

        try:
            response = requests.get(full_url, timeout=10)

            if response.status_code == 404:
                return jsonify(
                    {"success": False, "error": "The agent cannot comprehend your query", "endpoint": endpoint}
                )

            if response.status_code != 200:
                return jsonify(
                    {"success": False, "error": "An error occurred while processing your query", "endpoint": endpoint}
                )

            api_data = response.json()

            # Determine result type and structure data for frontend formatting
            result_type = "unknown"
            data = None
            total_results = 0

            if "/count" in endpoint:
                result_type = "count"
                data = {"total_bridges": api_data.get("total_bridges", 0)}
                total_results = 1

            elif "/group-by-county" in endpoint:
                result_type = "group_by_county"
                data = api_data
                total_results = len(api_data)

            elif "/group-by-spans" in endpoint:
                result_type = "group_by_spans"
                data = api_data
                total_results = len(api_data)

            elif "/search" in endpoint or "/bridges" in endpoint:
                result_type = "bridges"
                results = api_data.get("results", api_data if isinstance(api_data, list) else [])
                total_results = api_data.get("count", len(results))
                # Limit to 100 for display
                data = results[:100]

            elif "/inspections" in endpoint:
                result_type = "inspections"
                results = api_data if isinstance(api_data, list) else []
                total_results = len(results)
                data = results[:50]

            else:
                result_type = "raw"
                data = api_data
                total_results = 1

            return jsonify(
                {
                    "success": True,
                    "result_type": result_type,
                    "data": data,
                    "endpoint": endpoint,
                    "total_results": total_results,
                }
            )

        except requests.RequestException as e:
            logger.error(f"Error calling API endpoint {endpoint}: {e}")
            return (
                jsonify(
                    {"success": False, "error": "Connection error while processing your query", "endpoint": endpoint}
                ),
                500,
            )

    except Exception as e:
        logger.error(f"Error in agent_query: {e}")
        return jsonify({"success": False, "error": "An unexpected error occurred"}), 500


@app.route("/api/agent/examples")
def agent_examples():
    """Get example queries for the AI agent."""
    examples = [
        "How many bridges are in the database?",
        "Show me bridges in Westchester county",
        "Find bridges that carry highways",
        "What bridges cross rivers?",
        "Show me bridges with more than 3 spans",
        "Find all bridges in Orange county that have at least 2 spans",
        "Show me recent inspections",
        "What are the bridge statistics by county?",
    ]
    return jsonify({"examples": examples})


# ============================================================================
# STATIC PAGES
# ============================================================================


@app.route("/")
def index():
    """Serve the main application."""
    return render_template("index.html")


# ============================================================================
# BRIDGE ENDPOINTS (READ-ONLY)
# ============================================================================


@app.route("/api/bridges/count")
def bridges_count():
    """Get total count of bridges."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM bridges")
            count = cursor.fetchone()[0]
        return jsonify({"total_bridges": count})
    except Exception as e:
        logger.error(f"Error in bridges_count: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges")
def get_bridges():
    """Get paginated list of bridges."""
    try:
        limit = request.args.get("limit", 10, type=int)
        offset = request.args.get("offset", 0, type=int)

        query = "SELECT * FROM bridges LIMIT ? OFFSET ?"
        bridges = execute_bridge_query(query, (limit, offset))

        return jsonify(bridges)
    except Exception as e:
        logger.error(f"Error in get_bridges: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges/search")
def search_bridges():
    """Search bridges with multiple filters."""
    try:
        bin_id = request.args.get("bin")
        county = request.args.get("county")
        carried = request.args.get("carried")
        crossed = request.args.get("crossed")
        min_spans = request.args.get("min_spans", type=int)
        max_spans = request.args.get("max_spans", type=int)
        spans = request.args.get("spans", type=int)

        sort_by = request.args.get("sort", "id")
        order = request.args.get("order", "asc")
        limit = request.args.get("limit", type=int)

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

        if spans is not None:
            query += " AND spans = ?"
            params.append(spans)

        if min_spans:
            query += " AND spans >= ?"
            params.append(min_spans)

        if max_spans:
            query += " AND spans <= ?"
            params.append(max_spans)

        # Safe sort handling
        VALID_SORTS = {
            "id": "id",
            "bin": "bin",
            "spans": "spans",
            "carried": "carried",
            "crossed": "crossed",
            "county": "county",
        }
        VALID_ORDERS = {"asc": "ASC", "desc": "DESC"}

        sort_col = VALID_SORTS.get(sort_by, "id")
        sort_order = VALID_ORDERS.get(order.lower(), "ASC")
        query += f" ORDER BY {sort_col} {sort_order}"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        bridges = execute_bridge_query(query, params)

        return jsonify({"count": len(bridges), "results": bridges})
    except Exception as e:
        logger.error(f"Error in search_bridges: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges/by-carried/<feature>")
def bridges_by_carried(feature):
    """Get bridges that carry a specific feature."""
    try:
        query = "SELECT * FROM bridges WHERE carried LIKE ?"
        bridges = execute_bridge_query(query, (f"%{feature}%",))

        return jsonify({"carried": feature, "count": len(bridges), "results": bridges})
    except Exception as e:
        logger.error(f"Error in bridges_by_carried: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges/by-crossed/<feature>")
def bridges_by_crossed(feature):
    """Get bridges that cross a specific feature."""
    try:
        query = "SELECT * FROM bridges WHERE crossed LIKE ?"
        bridges = execute_bridge_query(query, (f"%{feature}%",))

        return jsonify({"crossed": feature, "count": len(bridges), "results": bridges})
    except Exception as e:
        logger.error(f"Error in bridges_by_crossed: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges/by-county/<county_name>")
def bridges_by_county(county_name):
    """Get bridges in a specific county."""
    try:
        county_num = None
        for num, name in COUNTY_MAP.items():
            if name.lower() == county_name.lower():
                county_num = num
                break

        if not county_num:
            return (
                jsonify({"error": f"Invalid county. Valid options: {', '.join(COUNTY_MAP.values())}"}),
                400,
            )

        sort_by = request.args.get("sort", "id")
        order = request.args.get("order", "asc")
        limit = request.args.get("limit", type=int)

        VALID_SORTS = {"id": "id", "bin": "bin", "spans": "spans", "carried": "carried", "crossed": "crossed"}
        VALID_ORDERS = {"asc": "ASC", "desc": "DESC"}

        sort_col = VALID_SORTS.get(sort_by, "id")
        sort_order = VALID_ORDERS.get(order.lower(), "ASC")

        query = f"SELECT * FROM bridges WHERE county = ? ORDER BY {sort_col} {sort_order}"
        params = [county_num]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        bridges = execute_bridge_query(query, params)

        return jsonify({"county": county_name, "count": len(bridges), "results": bridges})
    except Exception as e:
        logger.error(f"Error in bridges_by_county: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges/by-spans/<int:spans>")
def bridges_by_spans(spans):
    """Get bridges with a specific number of spans."""
    try:
        query = "SELECT * FROM bridges WHERE spans = ?"
        bridges = execute_bridge_query(query, (spans,))

        return jsonify({"spans": spans, "count": len(bridges), "results": bridges})
    except Exception as e:
        logger.error(f"Error in bridges_by_spans: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges/group-by-county")
def group_by_county():
    """Get bridge statistics grouped by county."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT county, COUNT(*) as count, AVG(spans) as avg_spans
                FROM bridges
                GROUP BY county
                ORDER BY county
            """
            )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            result = dict(row)
            result["county"] = map_county(result["county"])
            results.append(result)

        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in group_by_county: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/bridges/group-by-spans")
def group_by_spans():
    """Get bridge statistics grouped by span count."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT spans, COUNT(*) as count
                FROM bridges
                GROUP BY spans
                ORDER BY spans
            """
            )
            rows = cursor.fetchall()

        results = [dict(row) for row in rows]
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in group_by_spans: {e}")
        return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# INSPECTION ENDPOINTS
# ============================================================================


@app.route("/api/inspections/create", methods=["POST"])
def create_inspection():
    """Create a new bridge inspection."""
    try:
        data = request.json
        bin_id = data.get("bin")
        inspection_date = data.get("inspection_date")
        inspection_time = data.get("inspection_time")
        weather = data.get("weather")
        notes = data.get("notes")
        photos = data.get("photos")

        if not bin_id or not inspection_date or not inspection_time:
            return jsonify({"error": "Missing required fields"}), 400

        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM bridges WHERE bin = ?", (bin_id,))
            result = cursor.fetchone()

            if not result:
                return jsonify({"error": "Bridge not found"}), 404

            bridge_id = result[0]

            cursor.execute(
                """
                INSERT INTO inspections (bridge_id, bin, inspection_date, inspection_time, weather, notes, photos)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    bridge_id,
                    bin_id,
                    inspection_date,
                    inspection_time,
                    weather,
                    notes,
                    photos,
                ),
            )

            conn.commit()

        return jsonify({"success": True, "message": "Inspection logged successfully"})
    except Exception as e:
        logger.error(f"Error in create_inspection: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/inspections/<bin_id>")
def get_inspections(bin_id):
    """Get all inspections for a specific bridge."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, bin, inspection_date, inspection_time, weather, notes, photos, created_at
                FROM inspections
                WHERE bin = ?
                ORDER BY inspection_date DESC, inspection_time DESC
            """,
                (bin_id,),
            )

            rows = cursor.fetchall()

        inspections = []
        for row in rows:
            inspection = dict(row)
            if inspection.get("inspection_time"):
                inspection["inspection_time"] = str(inspection["inspection_time"])
            if inspection.get("inspection_date"):
                inspection["inspection_date"] = str(inspection["inspection_date"])
            inspections.append(inspection)

        return jsonify({"count": len(inspections), "inspections": inspections})
    except Exception as e:
        logger.error(f"Error in get_inspections: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/inspections")
def get_all_inspections():
    """Get all inspections across all bridges."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, bin, inspection_date, inspection_time, weather, notes, photos, created_at
                FROM inspections
                ORDER BY inspection_date DESC, inspection_time DESC
            """
            )

            rows = cursor.fetchall()

        inspections = []
        for row in rows:
            inspection = dict(row)
            if inspection.get("inspection_time"):
                inspection["inspection_time"] = str(inspection["inspection_time"])
            if inspection.get("inspection_date"):
                inspection["inspection_date"] = str(inspection["inspection_date"])
            inspections.append(inspection)

        return jsonify({"count": len(inspections), "inspections": inspections})
    except Exception as e:
        logger.error(f"Error in get_all_inspections: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/inspections/bridges")
def get_bridges_with_inspections():
    """Get list of bridges that have inspections with pagination and filters."""
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        offset = (page - 1) * per_page

        from_date = request.args.get("from_date")
        to_date = request.args.get("to_date")
        limit = request.args.get("limit", type=int)

        where_conditions = []
        params = []

        if from_date:
            where_conditions.append("inspection_date >= ?")
            params.append(from_date)
        if to_date:
            where_conditions.append("inspection_date <= ?")
            params.append(to_date)

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        with get_db() as conn:
            cursor = conn.cursor()

            count_query = f"""
                SELECT COUNT(DISTINCT bin) as total
                FROM inspections
                {where_clause}
            """
            cursor.execute(count_query, params)
            total_result = cursor.fetchone()
            total_bridges = total_result["total"] if total_result else 0

            main_query = f"""
                SELECT
                    bin,
                    COUNT(*) as inspection_count,
                    MAX(inspection_date) as latest_inspection_date,
                    MAX(created_at) as latest_created_at
                FROM inspections
                {where_clause}
                GROUP BY bin
                ORDER BY latest_inspection_date DESC, latest_created_at DESC
            """

            main_params = list(params)
            if limit:
                main_query += " LIMIT ?"
                main_params.append(limit)
            else:
                main_query += " LIMIT ? OFFSET ?"
                main_params.extend([per_page, offset])

            cursor.execute(main_query, main_params)
            rows = cursor.fetchall()

        bridges = []
        for row in rows:
            bridge = dict(row)
            if bridge.get("latest_inspection_date"):
                bridge["latest_inspection_date"] = str(bridge["latest_inspection_date"])
            if bridge.get("latest_created_at"):
                bridge["latest_created_at"] = str(bridge["latest_created_at"])
            bridges.append(bridge)

        if limit:
            total_pages = 1
            page = 1
        else:
            total_pages = (total_bridges + per_page - 1) // per_page if total_bridges > 0 else 0

        return jsonify(
            {
                "bridges": bridges,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_bridges": total_bridges,
                    "total_pages": total_pages,
                },
            }
        )
    except Exception as e:
        logger.error(f"Error in get_bridges_with_inspections: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/inspections/delete-by-bridge/<bin_id>", methods=["DELETE"])
def delete_inspections_by_bridge(bin_id):
    """Delete all inspections for a specific bridge."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM inspections WHERE bin = ?", (bin_id,))
            deleted_count = cursor.rowcount
            conn.commit()

        return jsonify(
            {
                "success": True,
                "message": f"Deleted {deleted_count} inspection(s) for bridge {bin_id}",
            }
        )
    except Exception as e:
        logger.error(f"Error in delete_inspections_by_bridge: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/inspections/delete-all", methods=["DELETE"])
def delete_all_inspections():
    """Delete all inspections (use with caution)."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM inspections")
            deleted_count = cursor.rowcount
            conn.commit()

        return jsonify({"success": True, "message": f"Deleted all {deleted_count} inspection(s)"})
    except Exception as e:
        logger.error(f"Error in delete_all_inspections: {e}")
        return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# PHOTO UPLOAD ENDPOINTS
# ============================================================================


@app.route("/api/inspections/upload-photo", methods=["POST"])
def upload_photo():
    """Upload a photo to local filesystem storage."""
    try:
        if "photo" not in request.files:
            return jsonify({"error": "No photo file provided"}), 400

        file = request.files["photo"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        file_extension = os.path.splitext(file.filename)[1].lower()
        blob_name = f"{uuid.uuid4()}{file_extension}"

        file_path = os.path.join(LOCAL_UPLOAD_DIR, blob_name)
        file.save(file_path)

        photo_url = f"/uploads/{blob_name}"
        return jsonify({"success": True, "url": photo_url, "blobName": blob_name})

    except Exception as e:
        logger.error(f"Photo upload failed: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/inspections/delete-photo", methods=["DELETE"])
def delete_photo():
    """Delete a photo from local filesystem storage."""
    try:
        data = request.json
        blob_name = data.get("blobName")

        if not blob_name:
            return jsonify({"error": "No blob name provided"}), 400

        file_path = os.path.join(LOCAL_UPLOAD_DIR, blob_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            return jsonify({"error": "Photo not found"}), 404

        return jsonify({"success": True, "message": "Photo deleted successfully"})

    except Exception as e:
        logger.error(f"Error deleting photo: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded files from local storage."""
    try:
        return send_from_directory(LOCAL_UPLOAD_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route("/api/inspections/proxy-photo", methods=["POST"])
def proxy_photo():
    """Proxy endpoint to fetch photos (or serve local files)."""
    try:
        data = request.json
        photo_url = data.get("url")

        if not photo_url:
            return jsonify({"error": "No URL provided"}), 400

        if photo_url.startswith("/uploads/"):
            filename = photo_url.split("/")[-1]
            return send_from_directory(LOCAL_UPLOAD_DIR, filename)

        response = requests.get(photo_url, timeout=10)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch photo"}), 500

        return Response(
            response.content,
            mimetype=response.headers.get("Content-Type", "image/jpeg"),
            headers={
                "Content-Disposition": "attachment",
                "Access-Control-Allow-Origin": "*",
            },
        )

    except Exception as e:
        logger.error(f"Error proxying photo: {e}")
        return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
