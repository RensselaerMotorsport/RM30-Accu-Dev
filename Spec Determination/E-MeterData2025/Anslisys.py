import numpy as np
import matplotlib.pyplot as plt
import nptdms as tdms
import os
import pandas as pd
from dataclasses import dataclass
from pickle import dump, load
import warnings as warn
import time
from Construct_Repaired_Teams import Team, EMeterData

Teams = load(open("Teams.pickle", "rb"))
EnduranceRuns = load(open("Endurance_Runs.pickle", "rb"))

TeamsThatFinish = [
    "San Jose State",
    "Polytechnique Montreal",
    "Oregon State",
    "Pittsburgh",
    "Georgia Tech",
    "Washington",
    "RIT",
    "U Conn.",
    "Purdue",
    "NUS",
    "Cincinnati",
    "UC Santa Cruz",
    "MIT",
    "Alberta",
    "Auburn",
    "Laval",
    "Missouri S&T",
    "Toronto"]

if __name__ == "__main__":
#     print("Running additional analysis...")
#     # Generating CSVs for each Endurance run by MIT
    # Create Excel file with a separate sheet for each team's entire endurance run data
#     # Generating CSVs for each Autocross run by MIT
    for Teamt in TeamsThatFinish:
        for i in Teams[Teamt].get_runs_by_event("ENDUR"):
              i.to_csv(f"FinishedEndurances/{i.name.split('\\')[-1]}_Endurance_Run.csv")

# maxPower=0
# for i in EnduranceRuns.keys():
#     if maxPower < EnduranceRuns[i].MaxPower():
#         MaxPowerTeam = EnduranceRuns[i].team
#         maxPower = max(maxPower, EnduranceRuns[i].MaxPower())
# print(f"Max Power for Endurance Runs: {maxPower}")
# # print(f"Team with Max Power for Endurance Runs: {MaxPowerTeam}")
# StintsDict = {}
# for i in EnduranceRuns.keys():
#     StintsDict[EnduranceRuns[i].name] = EnduranceRuns[i].stints
# pd.DataFrame.from_dict(StintsDict, orient='index', columns=['Stints']).to_csv("Stints.csv")
