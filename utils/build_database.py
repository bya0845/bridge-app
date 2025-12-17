"""
Build SQLite database from CSV data for bundled deployment.
Run this script locally to create bridges.db, then commit it to the repository.
"""
import sqlite3
import csv
import os
from pathlib import Path

# Paths
CSV_FILE = "./data/bridge_data.csv"
DB_FILE = "./data/bridges.db"
INSPECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bridge_id INTEGER NOT NULL,
    bin VARCHAR(50) NOT NULL,
    inspection_date DATE NOT NULL,
    inspection_time TIME NOT NULL,
    weather VARCHAR(50),
    notes TEXT,
    photos TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bridge_id) REFERENCES bridges(id)
);
CREATE INDEX IF NOT EXISTS idx_inspections_bin_date ON inspections(bin, inspection_date);
"""

def create_database():
    """Create SQLite database from CSV file."""
    
    # Remove existing database
    if os.path.exists(DB_FILE):
        print(f"Removing existing database: {DB_FILE}")
        os.remove(DB_FILE)
    
    # Ensure data directory exists
    Path("./data").mkdir(parents=True, exist_ok=True)
    
    # Create connection
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    print(f"Creating database: {DB_FILE}")
    
    # Create bridges table
    print("Creating bridges table...")
    cursor.execute("""
        CREATE TABLE bridges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bin VARCHAR(50) UNIQUE,
            county INTEGER,
            carried VARCHAR(255),
            crossed VARCHAR(255),
            latitude REAL,
            longitude REAL,
            spans INTEGER
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_county ON bridges(county)")
    cursor.execute("CREATE INDEX idx_bin ON bridges(bin)")
    cursor.execute("CREATE INDEX idx_spans ON bridges(spans)")
    
    # Import CSV data
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: CSV file not found: {CSV_FILE}")
        print("Please ensure bridge_data.csv exists in the data/ directory")
        conn.close()
        return False
    
    print(f"Importing data from {CSV_FILE}...")
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            try:
                # Handle different column name variations
                bin_val = row.get('BIN') or row.get('bin') or row.get('id')
                county = int(row.get('County') or row.get('county') or 0)
                carried = row.get('Carried') or row.get('carried') or ''
                crossed = row.get('Crossed') or row.get('crossed') or ''
                
                # Handle latitude/longitude
                try:
                    lat = float(row.get('Latitude') or row.get('latitude') or 0)
                    lon = float(row.get('Longitude') or row.get('longitude') or 0)
                except (ValueError, TypeError):
                    lat = 0.0
                    lon = 0.0
                
                # Handle spans
                try:
                    spans = int(row.get('Spans') or row.get('spans') or 0)
                except (ValueError, TypeError):
                    spans = 0
                
                cursor.execute("""
                    INSERT INTO bridges (bin, county, carried, crossed, latitude, longitude, spans)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (bin_val, county, carried, crossed, lat, lon, spans))
                
                count += 1
                
                if count % 500 == 0:
                    print(f"  Imported {count} bridges...")
                    
            except Exception as e:
                print(f"  Warning: Skipped row due to error: {e}")
                continue
    
    conn.commit()
    print(f"[OK] Imported {count} bridges successfully")
    
    # Create inspections table
    print("Creating inspections table...")
    cursor.executescript(INSPECTIONS_SCHEMA)
    conn.commit()
    
    # Get database stats
    cursor.execute("SELECT COUNT(*) FROM bridges")
    bridge_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT county) FROM bridges")
    county_count = cursor.fetchone()[0]
    
    # Get file size
    db_size = os.path.getsize(DB_FILE) / (1024 * 1024)  # MB
    
    print("\n" + "="*60)
    print("[SUCCESS] Database created successfully!")
    print("="*60)
    print(f"File: {DB_FILE}")
    print(f"Size: {db_size:.2f} MB")
    print(f"Bridges: {bridge_count:,}")
    print(f"Counties: {county_count}")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the database: python test_database.py")
    print("2. Commit to repository: git add data/bridges.db")
    print("3. Deploy with your app (no cloud database needed!)")
    print("="*60)
    
    conn.close()
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Building SQLite Database for Bundled Deployment")
    print("="*60 + "\n")
    
    success = create_database()
    
    if not success:
        exit(1)

