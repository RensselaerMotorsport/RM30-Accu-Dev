import numpy as np
import matplotlib.pyplot as plt
import nptdms as tdms
import os
import pandas as pd
from dataclasses import dataclass
from pickle import dump, load
import warnings as warn
import time
from Construct_Repaired_Teams import Team, EMeterData, GenerateRunFilesForTeam, GenerateTeams, RepairEnduranceRuns

Teams = load(open("Teams.pickle", "rb"))
EnduranceRuns = load(open("Endurance_Runs.pickle", "rb"))
def plot_energy_usage_with_time(csv_path="EnergyUsage.csv"):
    """
    Plots a bar chart of energy usage (kWh) for each team, overlaid with a line graph of time (s).
    """
    df = pd.read_csv(csv_path, index_col=0)
    teams = df.index
    energy = df["EnergyUsage_kWh"]
    time_s = df["Time (s)"] if "Time (s)" in df.columns else None
    peak_power = df["PeakPower_kW"] if "PeakPower_kW" in df.columns else None

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for energy usage
    bars = ax1.bar(teams, energy, color="tab:blue", alpha=0.7, label="Energy Usage (kWh)")
    ax1.set_xlabel("Team")
    ax1.set_ylabel("Energy Usage (kWh)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticklabels(teams, rotation=45, ha="right")

    # Line graph for time (left axis)
    ax2 = ax1.twinx()
    if time_s is not None:
        ax2.plot(teams, time_s, color="tab:red", marker="o", label="Time (s)")
        ax2.set_ylabel("Time (s)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    # Line graph for peak power (third axis)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    if peak_power is not None:
        ax3.plot(teams, peak_power, color="tab:green", marker="x", linestyle="--", label="Peak Power (kW)")
        ax3.set_ylabel("Peak Power (kW)", color="tab:green")
        ax3.tick_params(axis="y", labelcolor="tab:green")
        ax3.set_ylim(bottom=0)

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="center right")

    plt.title("Energy Usage, Time, and Peak Power by Team")
    plt.tight_layout()
    plt.show()

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
    "Missouri S&T"]

if __name__ == "__main__":
    runsheets = {}
    # for run in Teams["MIT"].get_runs_by_event("ENDUR"):
    #     runsheets[run.name.split("\\")[-1]] = run.to_pandas()
    #     print(run.name)
    # with pd.ExcelWriter('SHS_Endurance_Runs.xlsx') as writer:
    #     for sheet_name, df in runsheets.items():
    #         print(f"Writing sheet: {sheet_name}")
    #         safe_sheet_name = sheet_name.replace(":", "_")
    #         df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    #     Example usage of the plotting function:
    # plot_energy_usage_with_time()
    # EnergyUsage = {}
    # for Teamt in TeamsThatFinish:
    #     max_power = None
    #     energy_val = None
    #     for run in Teams[Teamt].get_runs_by_event("ENDUR"):
    #         # Energy usage (take last run's value)
    #         energy_val = run.Energy[-1]
    #         # Peak power logic
    #         if hasattr(run, 'MaxPower'):
    #             run_power = run.MaxPower()
    #             if max_power is None or (run_power is not None and run_power > max_power):
    #                 max_power = run_power
    #     EnergyUsage[Teamt] = {
    #         'EnergyUsage_kWh': energy_val,
    #         'PeakPower_kW': max_power
    #     }
    # pd.DataFrame.from_dict(EnergyUsage, orient='index').to_csv("EnergyUsage.csv")

# maxPower = 0
# MaxPowerTeam = None
# StintsDict = {}
# for team_name in TeamsThatFinish:
#     # Find all endurance runs for this team
#     for run_key, run in EnduranceRuns.items():
#         if getattr(run, 'team', None) == team_name:
#             # MaxPower calculation
#             if maxPower < run.MaxPower():
#                 MaxPowerTeam = run.team
#                 maxPower = run.MaxPower()
#             # StintsDict population
#             StintsDict[run.name] = run.stints
# print(f"Max Power for Endurance Runs (TeamsThatFinish only): {maxPower}")
# # print(f"Team with Max Power for Endurance Runs: {MaxPowerTeam}")
# pd.DataFrame.from_dict(StintsDict, orient='index', columns=['Stints']).to_csv("Stints.csv")


