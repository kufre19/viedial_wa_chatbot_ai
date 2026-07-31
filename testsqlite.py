import sys

# 1. Check if the binary version is installed
try:
    import pysqlite3
    # 2. Mock the standard sqlite3 library with the new binary version
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

print("Python version:", sys.version)
print("-" * 40)

try:
    import sqlite3
    print("sqlite3 module: available")
    print("SQLite version:", sqlite3.sqlite_version)
    print("SQLite library path:", sqlite3.__file__)
    
    # Quick functional test
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("SELECT sqlite_version();")
    print("SQLite works: Yes (in-memory test passed)")
    conn.close()
    
except ImportError:
    print("sqlite3 module: NOT available")
except Exception as e:
    print("Error while testing SQLite:", e)