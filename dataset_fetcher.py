import pandas as pd
from datetime import datetime, timedelta, timezone
import os

os.makedirs("data", exist_ok=True)

end_date = datetime.now(timezone.utc).date()
start_date = end_date - timedelta(days=180)

url = (
    f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
    f"format=csv&starttime={start_date}&endtime={end_date}&minmagnitude=2.5"
)

df = pd.read_csv(url)
df.to_csv("data/earthquake_data.csv", index=False)

print("✅ Dataset saved as data/earthquake_data.csv")
