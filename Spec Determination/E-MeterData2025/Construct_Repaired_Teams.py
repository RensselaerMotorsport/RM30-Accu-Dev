import numpy as np
import matplotlib.pyplot as plt
import nptdms as tdms
import os
import pandas as pd
from dataclasses import dataclass
from pickle import dump, load
import warnings as warn
import time


#######################################################################################################
# UserInput:
VerboseDataset = True  # Set to True to print dataset information
DoGenerateRunFiles = True  # Set to True to generate RunFiles.pickle when none is found
ForceGenerateRunFiles = False  # Set to True to force generation of RunFiles.pickle even if it exists
VerboseRunFiles = True  # Set to True to print RunFiles processing information
Debug = False  # Set to True to print detailed RunFiles processing information
DoGenerateTeams = True  # Set to True to generate Teams.pickle when none is found
ForceGenerateTeams = True  # Set to True to force generation of Teams.pickle even if it exists
ReDump = True  # Set to True to re-dump Teams and RunFiles to pickle files
#######################################################################################################
WD = os.getcwd()
print("Working Directory:", WD)

TeamNames = {
    "201": "RIT",
    "202": "Polytechnique Montreal",
    "203": "Wisconsin",
    "204": "Georgia Tech",
    "205": "San Jose State",
    "206": "Cornell",
    "207": "Illinois",
    "208": "Washington",
    "209": "Pittsburgh",
    "210": "Toronto",
    "212": "NUS",
    "215": "Maryland",
    "216": "UC Santa Cruz",
    "217": "Columbia",
    "218": "UBC",
    "219": "Olin",
    "221": "NC State",
    "222": "Cincinnati",
    "223": "Northwestern",
    "224": "BYU",
    "225": "Western",
    "226": "Waterloo",
    "228": "McGill",
    "229": "Calgary",
    "230": "SDSU",
    "232": "RPI",
    "233": "Toronto Met",
    "234": "Penn",
    "235": "UCLA",
    "241": "UC Santa Barbara",
    "242": "Missouri S&T",
    "243": "Alberta",
    "244": "Nevada Reno",
    "245": "Texas Tech",
    "246": "Concordia",
    "248": "Kookmin",
    "250": "Binghamton",
    "253": "Cal Poly",
    "254": "Auburn",
    "256": "UTA",
    "257": "UC Berkeley",
    "258": "Texas A&M",
    "259": "Northeastern",
    "261": "McMaster",
    "262": "UC Davis",
    "263": "Minnesota",
    "265": "MIT",
    "268": "Laval",
    "270": "Purdue",
    "272": "ETS",
    "274": "Michigan",
    "275": "Oregon State",
    "279": "Penn State",
    "281": "Cal Poly Pomona",
    "282": "U of Florida",
    "285": "North Texas",
    "287": "UT Dallas",
    "289": "USC",
    "291": "Hawaii",
    "293": "North Dakota",
    "298": "Michigan Dearborn",
    "299": "CMU",
    "301": "U Conn."
    }

class EMeterData:
    def __init__(self, name, data, Team, event, ID, TempData=True, TeamSignalData=False):
        
        # Metadata
        self.name = name
        self.car_number = name[0:3]
        self.event = event
        self.dataID = ID
        self.team = Team
        self.repaired = False  # Flag to indicate if the run has been repaired
        self.stints = None  # Number of stints in the run, default is 1
        # Raw Data
        self.data = data

        # Parced Data
        self.SampleID = np.linspace(0, len(data) - 1, len(data))
        self.Voltage = data[:, 0]
        self.Current = data[:, 1]
        self.Energy = data[:, 2]
        self.GLV = data[:, 3]
        self.Violation = data[:, 4]
        if TempData == True:
            self.Temperature = data[:, 9]
        if TeamSignalData == True:
            self.TeamSignal1 = data[:, 5]
            self.TeamSignal2 = data[:, 6]
            self.TeamSignal3 = data[:, 7]
            self.TeamSignal4 = data[:, 8]
    def set_attribute_new(self, name, value):
        self.name = value
    
    def __repr__(self):
        return f"EMeterData(name={self.name}, data_shape={self.data.shape})"
    def append_data(self, new_data):
        #Combines two EMeterData objects by appending new data to the existing data
        if isinstance(new_data, EMeterData):
            #shift second energy figure
            new_data.Energy += self.Energy[-1]
            self.data = np.vstack((self.data, new_data.data))
            self.Voltage = self.data[:, 0]
            self.Current = self.data[:, 1]
            self.Energy = self.data[:, 2]
            self.GLV = self.data[:, 3]
            self.Violation = self.data[:, 4]
            if hasattr(new_data, 'Temperature'):
                self.Temperature = np.concatenate((self.Temperature, new_data.Temperature))
            if hasattr(new_data, 'TeamSignal1'):
                self.TeamSignal1 = np.concatenate((self.TeamSignal1, new_data.TeamSignal1))
                self.TeamSignal2 = np.concatenate((self.TeamSignal2, new_data.TeamSignal2))
                self.TeamSignal3 = np.concatenate((self.TeamSignal3, new_data.TeamSignal3))
                self.TeamSignal4 = np.concatenate((self.TeamSignal4, new_data.TeamSignal4))
        self.SampleID = np.linspace(0, len(self.data) - 1, len(self.data))
        self.repaired = True  # Set the repaired flag to True after appending data
    def to_csv(self, filename, Debug=True):
        if Debug:
            print(f"Saving EMeterData to {filename}")
            print(f"SampleID length: {len(self.SampleID)}")
            print(f"Voltage length: {len(self.Voltage)}")
            print(f"Current length: {len(self.Current)}")
            print(f"Energy length: {len(self.Energy)}")
            print(f"GLV length: {len(self.GLV)}")
            print(f"Power length: {len(self.Power())}")
        df = pd.DataFrame({
            'SampleID': self.SampleID,
            'Voltage': self.Voltage,
            'Current': self.Current,
            'Energy': self.Energy,
            'GLV': self.GLV,
            'Power': self.Power(),
        })
        df.to_csv(filename, index=False)
    
    def MaxVoltage(self):
        return np.max(self.Voltage)
    def MaxCurrent(self):
        return np.max(self.Current)
    def MaxEnergy(self):
        return np.max(self.Energy)
    def Regen(self):
        return (np.min(self.Current) < -10 and np.max(self.Current) > 10) # Assuming regen is defined as any negative current
    def Power(self):
        return self.Voltage * self.Current
    def MaxPower(self):
        return np.abs(np.max(self.Power()))
    def MaxTemperature(self):
        return np.max(self.Temperature)
    def MinTemperature(self):
        return np.min(self.Temperature)
    def DeltaT(self):
        return self.MaxTemperature() - self.MinTemperature()
    def minCurrent(self):
        return np.min(self.Current)
    def ViolationFlag(self):
        return np.any(self.Violation > 0)
    def Plot(self, Parameters=None):
        plt.figure(figsize=(12, 8))
        plt.plot(self.SampleID, self.Voltage, label='Voltage (V)', color='blue')
        plt.plot(self.SampleID, self.Current, label='Current (A)', color='orange')
        plt.plot(self.SampleID, self.Energy, label='Energy (J)', color='green')
        plt.plot(self.SampleID, self.GLV, label='GLV', color='red')
        if hasattr(self, 'Temprature'):
            plt.plot(self.SampleID, self.Temprature, label='Temperature (°C)', color='purple')
        plt.xlabel('Sample ID')
        plt.ylabel('Values')
        plt.title(f'EMeter Data for {self.name}')
        plt.legend()
        plt.grid()
        plt.show()
    def TempGraditent(self):
        if hasattr(self, 'Temprature'):
            return np.gradient(self.Temperature)
        else:
            raise AttributeError("Temperature data not available. Set TempData=True when initializing EMeterData.")
    def TempGraditentPlot(self):
        if hasattr(self, 'Temprature'):
            plt.figure(figsize=(12, 6))
            plt.plot(self.SampleID, self.TempGraditent(), label='Temperature Gradient', color='purple')
            plt.xlabel('Sample ID')
            plt.ylabel('Temperature Gradient (°C/sample)')
            plt.title(f'Temperature Gradient for {self.name}')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            raise AttributeError("Temperature data not available. Set TempData=True when initializing EMeterData.")
    def MaxTempGraditent(self):
        if hasattr(self, 'Temprature'):
            return np.max(self.TempGraditent())
        else:
            raise AttributeError("Temperature data not available. Set TempData=True when initializing EMeterData.")
    def get_runs_by_key(self, Keys, Name=False):
        filtered_runs = []
        Keys = Keys.keys()
        for key in Keys:
            if type(Keys[key]) != tuple:
                if getattr(self, key) == Keys[key]:
                    filtered_runs.append(self)
            if type(Keys[key]) == tuple:
                if getattr(self, key) >= Keys[key][0] and getattr(self, key) <= Keys[key][1]:
                    filtered_runs.append(self)
        if Name:
            filtered_runs = [run.name for run in filtered_runs]
        return filtered_runs

class Team:
    def __init__(self, name, team_data=[]):
        self.name = name
        self.team_data = team_data  # This should be a list of EMeterData objects for this team
    def add_data(self, emeter_data):
        if isinstance(emeter_data, EMeterData):
            self.team_data.append(emeter_data)
        else:
            raise TypeError("Expected an instance of EMeterData")
    def get_runs(self):
        return self.team_data
    def get_run(self, index):
        return self.team_data[index]
    def get_runs_by_event(self, event_name, Name=False, singular=False):
        if Name:
            return [run.name for run in self.team_data if run.event == event_name]
        return [run for run in self.team_data if run.event == event_name]
    def __repr__(self):
        return f"Team(name={self.name}, runs={len(self.team_data)})"
    def max_voltage(self):
            return max(run.MaxVoltage() for run in self.team_data)
    def max_current(self):
            return max(run.MaxCurrent() for run in self.team_data)
    def max_energy(self):
            return max(run.MaxEnergy() for run in self.team_data)
    def regen(self):
            return any(run.Regen() for run in self.team_data)
    def max_power(self):
            return max(run.MaxPower() for run in self.team_data)
    def get_runs_by_key(self, Dict, Name=False, Singular=False, Debug=False):
        filtered_runs = []
        Keys = list(Dict.keys())
        for run in self.team_data:
            for key in Keys:
                if Debug:
                    print(f"Checking key: {key} with value: {Dict[key]} for run: {run.name}")
                if type(Dict[key]) != tuple:
                    if getattr(run,key) == Dict[key]:  
                        if run not in filtered_runs:
                            filtered_runs.append(run)
                if type(Dict[key]) == tuple:
                    if getattr(run,key) >= Dict[key][0] and getattr(run,key) <= Dict[key][1]:
                        if run not in filtered_runs:
                            filtered_runs.append(run)  
        if Name:
            filtered_runs = [run.name for run in filtered_runs]  
        if Singular:
            if len(filtered_runs) > 1:
                raise ValueError("More than one run found with the given keys. Use Singular=False to return all runs.")
            elif len(filtered_runs) == 0:
                raise ValueError("No run found with the given keys.")
            return filtered_runs[0]       
        return filtered_runs
    def get_run_by_ID(self, ID, Name=False):
        for run in self.team_data:
            if run.dataID == ID:
                if Name:
                    return run.name
                return run
        raise ValueError(f"No run found with ID: {ID}")
    def remove_run_by_ID(self, ID):
        for run in self.team_data:
            if run.dataID == ID:
                self.team_data.remove(run)
                return
        raise ValueError(f"No run found with ID: {ID}")
    def to_csv(self, filename):
        all_data = []
        for run in self.team_data:
            run_data = {
                'SampleID': run.SampleID,
                'Voltage': run.Voltage,
                'Current': run.Current,
                'Energy': run.Energy,
                'GLV': run.GLV,
                'Violation': run.Violation,
                'TeamSignal1': run.TeamSignal1,
                'TeamSignal2': run.TeamSignal2,
                'TeamSignal3': run.TeamSignal3,
                'TeamSignal4': run.TeamSignal4,
                'Temperature': run.Temprature
            }
            all_data.append(run_data)
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
    
# # Load the TDMS file & Create EMeterData objects
# # and store them in a dictionary with the file path as the key
# and pickle them for later use
def GenerateRunFiles(VerboseRunFiles):
    if VerboseRunFiles:
        print("Generating RunFiles from TDMS files in Unzipped_TDMS directory")
    RunFiles = {}
    Failures = []
    for fldr in os.listdir(r"Unzipped_TDMS"):
        if VerboseRunFiles:
            print(f"Processing folder: {fldr}")
        for i in os.listdir(r"Unzipped_TDMS/" + fldr):
            if i.endswith('EV.tdms'):
                path = os.path.join(r"Unzipped_TDMS", fldr, i)
                RunName = path[:-5]
                RunEvent = i.split('_')[-1].split("-")[0][1:]
                RunID = i.split('_')[-2]
                if Debug:
                    print(path)
                TDMSTemp = tdms.TdmsFile.read(path).as_dataframe().to_numpy()
                if (TDMSTemp.shape[1] != 10 and TDMSTemp.shape[1] != 9):
                    Failures.append(path)
                    if VerboseRunFiles:
                        print("*************************\n"+f"Failure: {path} - Expected 10 or 9 columns, found {TDMSTemp.shape[1]}"+"\n*************************")
                    continue
                elif (TDMSTemp.shape[1] == 9 ):
                      # Extract event name from the file name

                    RunFiles[RunName] = EMeterData(
                        name=RunName,  
                        data=TDMSTemp,
                        Team=TeamNames[fldr[0:3]],  # Assuming the first 3 characters of the folder name are the team number
                        event=RunEvent,  
                        ID=RunID,
                        TempData=False,  # No temperature data in 9 columns
                    )
                else: 
                    RunFiles[RunName] = EMeterData(
                        name=RunName,  
                        data=TDMSTemp,
                        Team=TeamNames[fldr[0:3]],  # Assuming the first 3 characters of the folder name are the team number
                        event=RunEvent, 
                        ID=RunID
                        
                    )
                if Debug:
                    print("Name = ", RunName)
                    print("Team = ", fldr)
                    print("Event = ", RunEvent)
                    print("ID = ", RunID)
                if RunFiles[RunName].ViolationFlag():
                    Failures.append(path)
                    if VerboseRunFiles:
                        print("*************************\n"+f"Failure: {path} - Violation flag is set"+"\n*************************")
                    del RunFiles[RunName]  # Remove the entry if it has a violation flag
    if VerboseRunFiles:
        print("RunFiles generated successfully")
        print(f"Total RunFiles: {len(RunFiles)}")
        print(f"Failures found: {len(Failures)}")
    if len(Failures) > 0:
        with open("Failures.txt", "w") as f:   
            for failure in Failures:
                f.write(f"{failure}\n")
        
    #sorting RunFiles by team name and then by event name
    print("Sorting RunFiles by team name and then by event name")
    RunFiles = dict(sorted(RunFiles.items(), key=lambda item: (item[1].team, item[1].event)))
    print("Saving RunFiles and Failures to pickle files")
    dump(RunFiles, open("RunFiles.pickle", "wb"))
    dump(Failures, open("Failures.pickle", "wb"))
    return RunFiles, Failures
def load_run_files():
    if os.path.exists("RunFiles.pickle") and os.path.exists("Failures.pickle"):
        if VerboseRunFiles:
            print("Loading RunFiles.pickle")
        RunFiles = load(open("RunFiles.pickle", "rb"))
        Failures = load(open("Failures.pickle", "rb"))
    elif os.path.exists("RunFiles.pickle") and not os.path.exists("Failures.pickle"):
        warn.warn("Failures.pickle not found, but RunFiles.pickle exists.\n If this is unexpected check for any errors in the run files.")
        RunFiles = load(open("RunFiles.pickle", "rb"))
        Failures = {}
    elif not os.path.exists("RunFiles.pickle") and os.path.exists("Failures.pickle"):
        raise FileNotFoundError("RunFiles.pickle Please generate it first.")
    elif not os.path.exists("RunFiles.pickle") and not os.path.exists("Failures.pickle"):
        raise FileNotFoundError("RunFiles.pickle and Failures.pickle not found. Please generate them first.")
    if VerboseRunFiles:
        print("Done loading RunFiles.pickle")
    return RunFiles, Failures
def GenerateTeams(RunFiles):
    if VerboseRunFiles:
        print("Generating Teams from RunFiles")
    Teams = {}
    for i in RunFiles:
        team_name = RunFiles[i].team
        if team_name not in Teams:
            Teams[team_name] = Team(name=team_name, team_data=[])
        Teams[team_name].add_data(RunFiles[i])
    if VerboseRunFiles:
        print("Teams generated successfully")
        print(f"Total Teams: {len(Teams)}")
    return Teams
def load_teams():
    if os.path.exists("Teams.pickle"):
        if VerboseRunFiles:
            print("Loading Teams.pickle")
        Teams = load(open("Teams.pickle", "rb"))
    else:
        raise FileNotFoundError("Teams.pickle not found. Please generate it first.")
    return Teams
def StartupRunfiles():
    if os.path.exists("RunFiles.pickle") and not ForceGenerateRunFiles:
        if VerboseRunFiles:
            print("RunFiles.pickle found, loading it")
        RunFiles, Failures = load_run_files()
    elif DoGenerateRunFiles or ForceGenerateRunFiles:
        if VerboseRunFiles:
            print("Generating RunFiles.pickle")
        RunFiles, Failures = GenerateRunFiles(VerboseRunFiles)
    else:
        raise FileNotFoundError("RunFiles.pickle not found and DoGenerateRunFiles is False. Exiting.")
        exit(1)
    return RunFiles, Failures
def StartupTeams():
    Teams = {}
    if os.path.exists("Teams.pickle") and not ForceGenerateTeams:
        if VerboseRunFiles:
            print("Teams.pickle found, loading it")
        Teams = load_teams()
    elif DoGenerateTeams or ForceGenerateTeams:
        if VerboseRunFiles:
            print("Generating Teams.pickle")
        Teams = GenerateTeams(RunFiles)
    else:
        raise FileNotFoundError("Teams.pickle not found and DoGenerateTeams is False. Exiting.")
    return Teams
def DatasetOuptut(Teams, RunFiles):
    CountEnduranceRuns = 0
    CountAutocrossRuns = 0
    CountAccelerationRuns = 0
    CountSkidpadRuns = 0
    CountOthersRuns = 0
    CountRegen = 0
    CountRuns = 0
    for team in Teams.values():
        for run in team.get_runs():
            CountRuns += 1
            if Debug:
                print(run.event)
            if run.event == "ENDUR":
                CountEnduranceRuns += 1
            elif run.event == "AUTOX":
                CountAutocrossRuns += 1
            elif run.event == "ACCEL":
                CountAccelerationRuns += 1
            elif run.event == "SKID":
                CountSkidpadRuns += 1
            else:
                CountOthersRuns += 1
                #writing other runs to a file
                with open("OtherRuns.txt", "a") as f:
                    f.write(f"{run.name} Event ={run.event}\n")
        if team.regen():
            CountRegen += 1
    print("********************************************************************************\nDataset Information:\n")
    print(f"Number of runs: {len(RunFiles)}")
    print(f"Number of teams: {len(Teams)}") 
    print(f"Number of teams with regen: {CountRegen}\n")
    print(f"Number of Endurance runs: {CountEnduranceRuns}")
    print(f"Number of Autocross runs: {CountAutocrossRuns}")
    print(f"Number of Acceleration runs: {CountAccelerationRuns}")  
    print(f"Number of Skidpad runs: {CountSkidpadRuns}")
    print(f"Number of Other runs: {CountOthersRuns}")
    print("********************************************************************************\n")
def RepairEnduranceRuns(Team):
    if Debug:
        print("Repairing Endurance run for team:", Team.name)
    TeamEnduranceRuns = Team.get_runs_by_event("ENDUR")
    if len(TeamEnduranceRuns) == 0:
        warn.warn(f"No Endurance runs found for team {Team.name}. Skipping repair.")
        return None
    IDs = []
    for run in TeamEnduranceRuns:
        IDs.append(getattr(run, 'dataID'))
        if Debug:
            print(f"Run {run.name} has ID: {IDs[-1]}")
    if Debug:
        print("IDs found:", IDs)
    IDs = sorted(IDs, key=lambda x: int(x.split('-')[1]))
    if Debug:
        print("Sorted IDs:", IDs)
    Team.get_run_by_ID(IDs[0]).repaired = True  # Mark the first run as the repaired run
    for ID in IDs[1:]:
        Team.get_runs_by_key({"repaired":True},Singular=True).append_data(Team.get_run_by_ID(ID))
        Team.remove_run_by_ID(ID)
    Team.get_runs_by_key({"repaired":True},Singular=True).name = f"{Team.name}_ENDUR_Repaired"
    Team.get_runs_by_key({"repaired":True},Singular=True).stints = len(IDs)





    
# Main execution
if __name__ == "__main__":
    # Remove any existing Failures.txt file
    if os.path.exists("Failures.txt"):
        os.remove("Failures.txt")
    # Remove any existing OtherRuns.txt file
    if os.path.exists("OtherRuns.txt"):
        os.remove("OtherRuns.txt")
    # Startup RunFiles
    RunFiles, Failures = StartupRunfiles()
    # Startup Teams
    Teams = StartupTeams()
    # Reparing Endurance Runs
    for team in Teams.keys():
        RepairEnduranceRuns(Teams[team])
    # Create Pickle of Just Endurance Runs
    EnduranceRuns = {}
    for team in Teams.values():
        runs = team.get_runs_by_event("ENDUR")
        if runs:
            EnduranceRuns[team.name] = runs[0]
        else:
            warn.warn(f"No Endurance runs found for team {team.name}. Skipping.")
    dump(EnduranceRuns, open("Endurance_Runs.pickle", "wb"))
    # Save Teams & Runs to pickle file
    if ReDump:
        if VerboseRunFiles:
            print("Saving Teams and RunFiles to pickle files")
        dump(Teams, open("Teams.pickle", "wb"))
        dump(RunFiles, open("RunFiles.pickle", "wb"))
    # Print dataset information if VerboseDataset is True
    if VerboseDataset:
        DatasetOuptut(Teams, RunFiles)
    
    print("RunFiles and Teams are ready for analysis.")




