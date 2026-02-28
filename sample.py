import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Husain@1106",
        database="user_data_drug",
        port=3306
    )
    print("✅ MySQL connected successfully!")
except Exception as e:
    print("❌ Connection failed:", e)
