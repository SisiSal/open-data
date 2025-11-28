from statsbombpy import sb
import pandas as pd
import time

match_id = 19738
competition_id = 37
season_id = 4

# Get all matches for a specific competition (e.g., La Liga 2020/21)
# You need the competition_id and season_id first
matches = sb.matches(competition_id, season_id)

# Filter for your specific match ID
my_match = matches[matches['match_id'] == match_id]

# View the info
print(my_match[['match_date', 'home_team', 'away_team', 'home_score', 'away_score']])