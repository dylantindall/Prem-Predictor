import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV into Pandas dataframe
df = pd.read_csv("prem_results.csv")

conn = sqlite3.connect(":memory:")
df.to_sql("MATCHES", conn, index=False, if_exists="replace")

# Query here using SQL command
query = """
SELECT season, month FROM MATCHES
WHERE month = "August" and season = "2019-2020"
"""
results = pd.read_sql(query, conn)

print(results)







# # Histogram plot for Matchweeks against total games played.
# query = """
# SELECT match_week, COUNT(*) AS games_played
# FROM matches
# GROUP BY match_week

# """
# mw_vs_totalgames = pd.read_sql(query, conn)
# plt.figure(figsize=(10, 6))
# plt.bar(mw_vs_totalgames["match_week"], mw_vs_totalgames["games_played"], color="steelblue", edgecolor="black")
# plt.xlabel("Matchweek")
# plt.ylabel("Games Played")
# plt.title("Games per Matchweek")
# plt.xticks(mw_vs_totalgames["match_week"])  # show all matchweeks on x-axis
# plt.tight_layout()
# plt.show()