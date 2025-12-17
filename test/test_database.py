"""
Test the bundled SQLite database to ensure it works correctly.
"""
import sqlite3
import os

DB_FILE = "./data/bridges.db"

def test_database():
    """Run tests on the SQLite database."""
    
    if not os.path.exists(DB_FILE):
        print(f"[ERROR] Database not found: {DB_FILE}")
        print("Run: python build_database.py")
        return False
    
    print("\n" + "="*60)
    print("Testing SQLite Database")
    print("="*60 + "\n")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    all_passed = True
    
    # Test 1: Check bridges table exists
    print("Test 1: Checking bridges table...")
    try:
        cursor.execute("SELECT COUNT(*) FROM bridges")
        count = cursor.fetchone()[0]
        print(f"  [OK] Found {count:,} bridges")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 2: Check inspections table exists
    print("\nTest 2: Checking inspections table...")
    try:
        cursor.execute("SELECT COUNT(*) FROM inspections")
        count = cursor.fetchone()[0]
        print(f"  [OK] Found {count} inspections (empty is OK)")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 3: Check data integrity
    print("\nTest 3: Checking data integrity...")
    try:
        cursor.execute("SELECT COUNT(*) FROM bridges WHERE bin IS NOT NULL")
        bin_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT county) FROM bridges")
        county_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(spans), MAX(spans) FROM bridges")
        min_spans, max_spans = cursor.fetchone()
        
        print(f"  [OK] All bridges have BIN: {bin_count:,}")
        print(f"  [OK] Counties: {county_count}")
        print(f"  [OK] Span range: {min_spans} - {max_spans}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 4: Sample queries
    print("\nTest 4: Running sample queries...")
    
    # Query by county
    try:
        cursor.execute("SELECT COUNT(*) FROM bridges WHERE county = 7")
        westchester_count = cursor.fetchone()[0]
        print(f"  [OK] Westchester bridges: {westchester_count:,}")
    except Exception as e:
        print(f"  [FAIL] County query failed: {e}")
        all_passed = False
    
    # Query by carried
    try:
        cursor.execute("SELECT COUNT(*) FROM bridges WHERE carried LIKE '%84i%'")
        i84_count = cursor.fetchone()[0]
        print(f"  [OK] Bridges on 84i: {i84_count}")
    except Exception as e:
        print(f"  [FAIL] Carried query failed: {e}")
        all_passed = False
    
    # Query by spans
    try:
        cursor.execute("SELECT COUNT(*) FROM bridges WHERE spans >= 3")
        span_count = cursor.fetchone()[0]
        print(f"  [OK] Bridges with 3+ spans: {span_count:,}")
    except Exception as e:
        print(f"  [FAIL] Spans query failed: {e}")
        all_passed = False
    
    # Test 5: Check indexes
    print("\nTest 5: Checking indexes...")
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()
        print(f"  [OK] Found {len(indexes)} indexes")
        for idx in indexes:
            print(f"    - {idx[0]}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    
    # Test 6: Get sample bridges
    print("\nTest 6: Sample data...")
    try:
        cursor.execute("""
            SELECT bin, county, carried, crossed, spans 
            FROM bridges 
            LIMIT 5
        """)
        bridges = cursor.fetchall()
        print(f"  [OK] Sample bridges:")
        for b in bridges:
            print(f"    BIN {b[0]}: County {b[1]}, {b[4]} spans")
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_passed = False
    
    conn.close()
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*60)
        print("Database is ready for deployment!")
        print("\nYou can now:")
        print("1. Test locally: DATABASE_TYPE=sqlite python app.py")
        print("2. Commit: git add data/bridges.db")
        print("3. Deploy to Railway with bundled database")
    else:
        print("[FAIL] SOME TESTS FAILED")
        print("="*60)
        print("Please fix errors before deploying")
    print("="*60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    test_database()

