# To download necessary packages, run:
# python -m pip install tqdm pandas numpy matplotlib
try:
    import pickle
    import multiprocessing as mp
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import os
    import datetime
    import CellSelection as CS
except ImportError as e:
    print(f"Error importing module: {e}. Please ensure all required packages are installed.")
    print("You can run \n'python -m pip install tqdm pandas numpy matplotlib'\n in command prompt to install missing packages.")
    raise

Locations = {
    "Cell": 0,
    "Pack Voltage (V)": 1,
    "SMod": 2,
    "S": 3,
    "Pack Current (A)": 4,
    "P": 5,
    "MCnt": 6,
    "Pack Power (kW)": 7,
    "Pack Energy (kWh)": 8,
    "Total Pack Weight (kg)": 9,
    "Cell Weight (kg)": 10,
    "Total Wall Weight (kg)": 11,
    "Grav Energy Density (Wh/kg)": 12,
    "Grav Power Density (W/kg)": 13,
    "Pack Length (mm)": 14,
    "Pack Width (mm)": 15,
    "Pack Height (mm)": 16,
    "Total Wall Volume (mm³)": 17
}

def normalize_cell_name(cellname, normalize=True):
    """Normalize cell names so all GRP variants are treated as one, and all SLPB variants are treated as one"""
    if not normalize:
        return cellname

    cell_str = str(cellname).strip().upper()  # Convert to string, strip whitespace, and uppercase for comparison

    if cell_str.startswith('GRP'):
        return 'GRP'
    if cell_str.startswith('SLPB'):
        return 'SLPB'
    else:
        return cell_str

def DoCellOptimization(Configs, OPTPower, OPTEnergy):
    valid_configs = []
    for k in Configs:
        power = float(k["Pack Power (kW)"])
        energy = float(k["Pack Energy (kWh)"])
        if power >= OPTPower and energy >= OPTEnergy:
            valid_configs.append(k)
    
    results = [None, None, None]
    used_cells = [None, None, None]

    for i in range(3):
        best_Weight=np.inf
        used_cells_norm = [normalize_cell_name(cell) for cell in used_cells if cell is not None]
        for config in valid_configs:
            current_cell = normalize_cell_name(config.iloc[Locations["Cell"]])
            current_weight = config.iloc[Locations["Total Pack Weight (kg)"]]
            if (current_cell not in used_cells_norm and
               best_Weight >= current_weight):

                best_Weight = current_weight
                results[i] = config
        used_cells[i] = results[i].iloc[Locations["Cell"]]

    return [OPTPower, OPTEnergy, {
        "results": results
     }]

def MakeArgs(configs, OptPWRRng=[60,120], OptPwrCnt = 10, OptEnergRng = [3,8], OptEnergCnt = 10):
    args = []
    for i in np.linspace(OptPWRRng[0], OptPWRRng[1], OptPwrCnt):
        for j in np.linspace(OptEnergRng[0], OptEnergRng[1], OptEnergCnt):
            args.append((configs, i, j))
    return args

def FilterConfigs(configs, MinVolt=0, MaxVolt=510, minPWR=60, maxPWR=np.inf, minEnerg=3, maxEnerg=np.inf, min_weight=0, max_weight=50):
    filtered = []
    required_columns = ["Pack Power (kW)", "Pack Energy (kWh)", "Total Pack Weight (kg)"]
    for col in required_columns:
        if col not in configs.columns:
            raise ValueError(f"Missing required column: {col}")
    # Running statistics
    stats = {
        "weight": {"min": None, "max": None, "sum": 0.0, "count": 0},
        "energy": {"min": None, "max": None, "sum": 0.0, "count": 0},
        "power": {"min": None, "max": None, "sum": 0.0, "count": 0}
    }
    for _, config in tqdm(configs.iterrows(), total=len(configs), desc="Filtering configs"):
        if (
            config["Pack Power (kW)"] >= minPWR and
            config["Pack Power (kW)"] <= maxPWR and
            config["Pack Energy (kWh)"] >= minEnerg and
            config["Pack Energy (kWh)"] <= maxEnerg and
            config["Total Pack Weight (kg)"] >= min_weight and
            config["Total Pack Weight (kg)"] <= max_weight and
            config["Voltage (V)"] >= MinVolt and
            config["Voltage (V)"] <= MaxVolt
        ):
            filtered.append(config)
            for key, col in zip(["weight", "energy", "power"], ["Total Pack Weight (kg)", "Pack Energy (kWh)", "Pack Power (kW)"]):
                val = float(config[col])
                if stats[key]["min"] is None or val < stats[key]["min"]:
                    stats[key]["min"] = val
                if stats[key]["max"] is None or val > stats[key]["max"]:
                    stats[key]["max"] = val
                stats[key]["sum"] += val
                stats[key]["count"] += 1
    summary = {
        key: {
            "min": stats[key]["min"],
            "max": stats[key]["max"],
            "mean": stats[key]["sum"] / stats[key]["count"] if stats[key]["count"] > 0 else None
        }
        for key in stats
    }
    return filtered, summary

def optimization_worker(args):
    return DoCellOptimization(*args)

if __name__ == "__main__":
    Density = 100
    if not os.path.exists("ConfigsAll.csv") or not os.path.exists("ConfigsCyl.csv"):
        print("Config files not found, generating...")
        CS.GenerateCellsForAnalisys()
    if os.path.exists(f"results{str(Density)}.pkl"):
        print(f"Found existing results{str(Density)}.pkl, loading...")
        resultsDict = pickle.load(open(f"results{str(Density)}.pkl", "rb"))
        resultsCylDict, resultsAllDict = resultsDict[0], resultsDict[1]
        print(len(resultsCylDict), len(resultsAllDict))
        allConfigs = pd.read_csv("ConfigsAll.csv")
        CylConfigs = pd.read_csv("ConfigsCyl.csv")
    else:
        resultsCylDict = {}
        resultsAllDict = {}
        Processes = mp.cpu_count() - 1
        print("Importing All Configs")
        allConfigs, allSummary = FilterConfigs(pd.read_csv("ConfigsAll.csv"))
        print(f"Survived filtration: {len(allConfigs)} configs.")
        if len(allConfigs) > 0:
            print(f"Weight: min={allSummary['weight']['min']:.2f}, max={allSummary['weight']['max']:.2f}, mean={allSummary['weight']['mean']:.2f}")
            print(f"Energy: min={allSummary['energy']['min']:.2f}, max={allSummary['energy']['max']:.2f}, mean={allSummary['energy']['mean']:.2f}")
            print(f"Power: min={allSummary['power']['min']:.2f}, max={allSummary['power']['max']:.2f}, mean={allSummary['power']['mean']:.2f}")
        print("Importing Cylindrical Configs")
        CylConfigs, cylSummary = FilterConfigs(pd.read_csv("ConfigsCyl.csv"))
        print(f"Survived filtration: {len(CylConfigs)} cylindrical configs.")
        if len(CylConfigs) > 0:
            print(f"Weight: min={cylSummary['weight']['min']:.2f}, max={cylSummary['weight']['max']:.2f}, mean={cylSummary['weight']['mean']:.2f}")
            print(f"Energy: min={cylSummary['energy']['min']:.2f}, max={cylSummary['energy']['max']:.2f}, mean={cylSummary['energy']['mean']:.2f}")
            print(f"Power: min={cylSummary['power']['min']:.2f}, max={cylSummary['power']['max']:.2f}, mean={cylSummary['power']['mean']:.2f}")
        print("Generating Args")
        argsAll = MakeArgs(allConfigs, OptPwrCnt=Density, OptEnergCnt=Density)
        print("Generating Args Cyl")
        argsCyl = MakeArgs(CylConfigs, OptPwrCnt=Density, OptEnergCnt=Density)
        pool = mp.Pool(processes=Processes)
        print("Starting Cylindrical")
        resultsCyl = list(tqdm(pool.imap(optimization_worker, argsCyl), total=len(argsCyl)))
        print("Cylindrical Done\nStarting All")
        resultsAll = list(tqdm(pool.imap(optimization_worker, argsAll), total=len(argsAll)))
        print("All Done")
        pool.close()
        pool.join()
        for i in resultsCyl:
            resultsCylDict[(i[0], i[1])] = i[2]
        for j in resultsAll:
            resultsAllDict[(j[0], j[1])] = j[2]
        results = [resultsCylDict, resultsAllDict]
        pickle.dump(results, open(f"results{str(Density)}.pkl", "wb"))
    print("Done")