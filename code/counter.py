import warnings
import pandas as pd
from statsbombpy import sb
from mplsoccer import Pitch

competitions = sb.competitions()

total_matches = 0
total_events = 0

for _, competition in competitions.iterrows():
    competition_id = competition['competition_id']
    season_id = competition['season_id']
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    for _, match in matches.iterrows():
        events = sb.events(match['match_id'])
        print(f"For match {match['match_id']} we have the following events")
        print(f"{len(events)}")
        total_events += len(events)
    print(f"{competition['competition_name']} ({competition['season_name']}): {len(matches)} matches")
    print(f"Competition ID: {competition_id}, season ID: {season_id}")
    total_matches += len(matches)

print(f"\nTotal matches available in StatsBomb dataset: {total_matches}")
print(f"\nTotal events available in StatsBomb dataset: {total_events}")

