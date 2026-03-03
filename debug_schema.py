import os
from dotenv import load_dotenv
import drug
import traceback

load_dotenv()
try:
    drug.ensure_schema()
    print("FINISHED RUNNING ENSURE SCHEMA")
except Exception as e:
    print("EXCEPTION IN ENSURE SCHEMA:", repr(e))
    traceback.print_exc()
