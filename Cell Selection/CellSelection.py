import numpy as np
import pandas as pd
import pickle
import torch
import os
import math
from tqdm import tqdm

############################################
CellsFilePath = "cells.csv"
WallDensity = 2.71e-6 # kg/mm^2
Verbose = False
USE_EXHAUSTIVE = True  # Set to False to use the original GenerateConfigs method
############################################

def Vprint(*args):
    if Verbose:
        print(*args)
    return None
    # Function implementation goes here

class Cell:
    def __init__(self, store, link, brand, model, package, capacity_mAh, current_A, weight_g, length_mm, thickness_mm, width_mm, voltage_V):
        self.store = store
        self.link = link
        self.brand = brand
        self.model = model
        self.package = package
        self.capacity_mAh = capacity_mAh
        self.current_A = current_A
        self.voltage_V = voltage_V
        self.weight_g = weight_g
        self.length_mm = length_mm
        if self.package == "Cylinder":
            self.diameter_mm = thickness_mm
            self.thickness_mm = None
            self.width_mm = None
        elif self.package == "Pouch":
            self.thickness_mm = thickness_mm
            self.width_mm = width_mm
            self.diameter_mm = None

    def volume(self):
        if self.package == "Cylinder":
            # Volume of a cylinder: πr²h
            radius = self.diameter_mm / 2
            return np.pi * (radius ** 2) * self.length_mm
        elif self.package == "Pouch":
            # Volume of a pouch cell: lwh
            return self.length_mm * self.thickness_mm * self.width_mm
        else:
            raise ValueError("Unknown cell package type")
    
    def bounding_box_vol(self):
        if self.package == "Cylinder":
            return self.length_mm * self.diameter_mm * self.diameter_mm
        elif self.package == "Pouch":
            return self.volume()
    
    def bounding_box(self):
        if self.package == "Cylinder":
            return (self.length_mm, self.diameter_mm, self.diameter_mm)
        elif self.package == "Pouch":
            return (self.length_mm, self.thickness_mm, self.width_mm)

    def to_csv(self, filename):
        df  = pd.DataFrame([{
            "Store": self.store,
            "Link": self.link,
            "Brand": self.brand,
            "Model": self.model,
            "Package": self.package,
            "Capacity_mAh": self.capacity_mAh,
            "Current_A": self.current_A,
            "Weight_g": self.weight_g,
            "Length_mm": self.length_mm,
            "Thickness_mm": self.thickness_mm,
            "Width_mm": self.width_mm,
            "Diameter_mm": self.diameter_mm,
            "Voltage_V": self.voltage_V
        }])
        df.to_csv(filename, index=False)

class Config:
    def __init__(self, cell, SMod, P, MCnt, ModWeight, WallDensity):
        self.cell = cell #Name of Cell
        self.SMod = SMod #Number of Series Cells Per Module
        self.P = P #Number of Parallel Cells Module
        self.MCnt = MCnt #Number of Modules
        self.S = SMod * MCnt #Total Number of Series Cells
        self.PackVoltage = self.cell.voltage_V * self.SMod * MCnt #Total Pack Voltage (V)
        self.CellWeight = ModWeight * MCnt / 1000 #Total Cell Weight (kg)
        self.Current = P * cell.current_A #Total Pack Current (A)
        self.Energy = self.PackVoltage * P *self.cell.capacity_mAh / 1000000 # in kWh
        self.Power = self.PackVoltage * self.Current / 1000 # in kW
        self.calc_module_dims(WallDensity)
        self.CalcTotalPackWeight()
    def calc_module_dims(self, WallDensity):
        Vprint("CalcDims")
        if self.cell.package == "Cylinder":
            Vprint("Cylindrical")
            self.ModL = self.cell.diameter_mm * self.SMod #Module Length (mm)
            self.ModH = self.cell.diameter_mm * self.P #Module Height (mm)
            self.ModWidth = self.cell.length_mm #Module Width (mm)
            self.InnerWallVolume = self.ModL * self.ModH * 2.3 #mm^3, assuming 2.3mm thicknes required for aluminum on vertical walls
            self.OuterWallVolume = (self.ModH * self.ModWidth * self.MCnt * 2 + self.ModL * self.ModH * 2) * 2.3 #mm^3, assuming 2.3mm thicknes required for aluminum on vertical walls
            self.RoofFloorVolume = self.ModL * self.ModWidth * self.MCnt * 2 * 3.2 #mm^3, assuming 3.2mm thicknes required for aluminum on roof and floor
            self.TotalWallVolume = self.InnerWallVolume * (self.MCnt - 1) + self.OuterWallVolume + self.RoofFloorVolume #total wall volume (mm^3)
            self.TotalWallWeight = self.TotalWallVolume * WallDensity #total wall weight (kg), assuming desnity above is in kg/mm^3
            self.PackLength = self.ModL #Pack Length (mm)
            self.PackWidth = self.ModWidth * self.MCnt #Pack Width (mm)
            self.PackHeight = self.ModH #Pack Height (mm)
        elif self.cell.package == "Pouch":
            Vprint("Pouch")
            self.ModL = self.cell.thickness_mm * self.SMod
            self.ModH = self.cell.length_mm
            self.ModWidth = self.cell.width_mm * self.P
            self.InnerWallVolume = self.ModL * self.ModH * 2.3 #mm^3
            self.OuterWallVolume = (self.ModH * self.ModWidth * self.MCnt * 2 + self.ModL * self.ModH * 2) * 2.3 #mm^3
            self.RoofFloorVolume = self.ModL * self.ModWidth * self.MCnt * 2 * 3.2 #mm^3
            self.TotalWallVolume = self.InnerWallVolume * (self.MCnt - 1) + self.OuterWallVolume + self.RoofFloorVolume
            self.TotalWallWeight = self.TotalWallVolume * WallDensity
            self.PackLength = self.ModL
            self.PackWidth = self.ModWidth * self.MCnt
            self.PackHeight = self.ModH
        Vprint("DimsDone")
    def CalcTotalPackWeight(self):
        self.TotalPackWeight = self.CellWeight + self.TotalWallWeight
        self.GravEnergyDensity = self.Energy / self.TotalPackWeight #kWh/kg
        self.GravPowerDensity = (self.Power / self.TotalPackWeight) if (self.Power <= 80000) else 80000/self.TotalPackWeight #kW/kg

    def to_csv(self, filename):
        df = pd.DataFrame([{
            "Cell":self.cell.Model,
            "SMod": self.SMod,
            "P": self.P,
            "MCnt": self.MCnt,
            "ModW": self.ModW,
            "PackVoltage":self.PackVoltage,
            "CCellWeight":self.CellWeight,
        }])
    def to_str(self):
        return f"{self.cell.Model}: {self.SMod}S{self.P}P, {self.MCnt} Modules, {self.PackVoltage}V, {self.CellWeight/1000:.1f}kg Cells + {self.TotalWallWeight/1000:.1f}kg Walls = {self.TotalPackWeight/1000:.1f}kg Total"

def GenerateConfig_CC(cell, MaxPower, TargetVoltage, ModuleMaxVoltage=100, ModuleMaxE=6000000, ModuleMaxCellWeight=8000, PotentialModuleCount=6):
    ViableConfigs = []
    current = MaxPower / (TargetVoltage)
    P = np.ceil(current / cell.current_A)
    Vprint(P, cell.model)
    if not isinstance(cell.voltage_V, (int, float)) or np.isnan(cell.voltage_V):
        Vprint(cell.model)
        Vprint(type(cell.voltage_V))
        Vprint(cell.voltage_V)
        breakpoint()
    for M in range(1, PotentialModuleCount + 1):
        # Determine the number of cells in series (S) to meet the target voltage
        SMod = int(np.ceil(TargetVoltage / (cell.voltage_V * M)))
        ModV = cell.voltage_V * SMod # Volts
        ModE = ModV * P * cell.capacity_mAh * 3.6 # Joules
        ModW = cell.weight_g * SMod * P # Grams
        Vprint(f"Trying {M} modules: {SMod}S, {P}P, {ModV}V, {ModE/1000000:.1f}MJ, {ModW:.1f}g")
        if ModV > ModuleMaxVoltage:
            continue
        elif ModE > ModuleMaxE:
            continue
        elif ModW > ModuleMaxCellWeight:
            continue
        elif ModV < 40:
            continue
        Vprint("passed")
        ViableConfigs.append(Config(cell, SMod, P, M, ModW, WallDensity=WallDensity))
    Vprint(type(ViableConfigs))
    return ViableConfigs if len(ViableConfigs) > 0 else None
def GenerateConfig_CF(cell, MaxPower, TargetVoltage, ModuleMaxVoltage=100, ModuleMaxE=6000000, ModuleMaxCellWeight=8000, PotentialModuleCount=6):
    ViableConfigs = []
    current = MaxPower / (TargetVoltage)
    P = np.ceil(current / cell.current_A)
    Vprint(P, cell.model)
    for M in range(1, PotentialModuleCount + 1):
        # Determine the number of cells in series (S) to meet the target voltage
        SMod = int(np.floor(TargetVoltage / (cell.voltage_V * M)))
        if SMod < 1:
            continue
        ModV = cell.voltage_V * SMod # Volts
        ModE = ModV * P * cell.capacity_mAh * 3.6 # Joules
        ModW = cell.weight_g * SMod * P # Grams
        Vprint(f"Trying {M} modules: {SMod}S, {P}P, {ModV}V, {ModE/1000000:.1f}MJ, {ModW:.1f}g")
        if ModV > ModuleMaxVoltage:
            continue
        elif ModE > ModuleMaxE:
            continue
        elif ModW > ModuleMaxCellWeight:
            continue
        elif ModV < 40:
            continue
        Vprint("passed")
        ViableConfigs.append(Config(cell, SMod, P, M, ModW, WallDensity=WallDensity))
    return ViableConfigs if len(ViableConfigs) > 0 else None
def GenerateConfig_FC(cell, MaxPower, TargetVoltage, ModuleMaxVoltage=100, ModuleMaxE=6000000, ModuleMaxCellWeight=8000, PotentialModuleCount=6):
    ViableConfigs = []
    current = MaxPower / (TargetVoltage)
    P = np.floor(current / cell.current_A)
    if P < 1:
        return None
    Vprint(P, cell.model)
    for M in range(1, PotentialModuleCount + 1):
        # Determine the number of cells in series (S) to meet the target voltage
        SMod = int(np.ceil(TargetVoltage / (cell.voltage_V * M)))
        ModV = cell.voltage_V * SMod # Volts
        ModE = ModV * P * cell.capacity_mAh * 3.6 # Joules
        ModW = cell.weight_g * SMod * P # Grams
        Vprint(f"Trying {M} modules: {SMod}S, {P}P, {ModV}V, {ModE/1000000:.1f}MJ, {ModW:.1f}g")
        if ModV > ModuleMaxVoltage:
            continue
        elif ModE > ModuleMaxE:
            continue
        elif ModW > ModuleMaxCellWeight:
            continue
        elif ModV < 40:
            continue
        Vprint("passed")
        ViableConfigs.append(Config(cell, SMod, P, M, ModW, WallDensity=WallDensity))
    return ViableConfigs if len(ViableConfigs) > 0 else None
def GenerateConfig_FF(cell, MaxPower, TargetVoltage, ModuleMaxVoltage=100, ModuleMaxE=6000000, ModuleMaxCellWeight=8000, PotentialModuleCount=6):
    ViableConfigs = []
    current = MaxPower / (TargetVoltage)
    P = np.floor(current / cell.current_A)
    if P < 1:
        return None
    Vprint(P, cell.model)
    for M in range(1, PotentialModuleCount + 1):
        # Determine the number of cells in series (S) to meet the target voltage
        SMod = int(np.floor(TargetVoltage / (cell.voltage_V * M)))
        if SMod < 1:
            continue
        ModV = cell.voltage_V * SMod # Volts
        ModE = ModV * P * cell.capacity_mAh * 3.6 # Joules
        ModW = cell.weight_g * SMod * P # Grams
        Vprint(f"Trying {M} modules: {SMod}S, {P}P, {ModV}V, {ModE/1000000:.1f}MJ, {ModW:.1f}g")
        if ModV > ModuleMaxVoltage:
            continue
        elif ModE > ModuleMaxE:
            continue
        elif ModW > ModuleMaxCellWeight:
            continue
        elif ModV < 40:
            continue
        # Vprint("passed")
        ViableConfigs.append(Config(cell, SMod, P, M, ModW, WallDensity=WallDensity))
    return ViableConfigs if len(ViableConfigs) > 0 else None

def GenerateConfig_Exhaustive(cell, MaxPower, ModuleMaxVoltage=100, ModuleMaxE=6000000, ModuleMaxCellWeight=8000, PotentialModuleCount=8):
    ViableConfigs = []
    for M in range(1, PotentialModuleCount + 1):
            for SMod in range(1, int(ModuleMaxVoltage / cell.voltage_V + 2)):
                denom = cell.current_A * SMod * cell.voltage_V
                if denom == 0 or denom != denom:  # denom != denom checks for NaN
                    continue
                max_p = MaxPower / denom
                if max_p != max_p or max_p <= 0:  # skip if max_p is NaN or non-positive
                    continue
                for P in range(1, int(max_p + 2)):
                    ModV = cell.voltage_V * SMod  # Volts
                    ModE = ModV * P * cell.capacity_mAh * 3.6  # Joules
                    ModW = cell.weight_g * SMod * P  # Grams
                    if (ModV > ModuleMaxVoltage or ModE > ModuleMaxE or ModW > ModuleMaxCellWeight):
                        continue
                    ViableConfigs.append(Config(cell, SMod, P, M, ModW, WallDensity=WallDensity))
    return ViableConfigs if len(ViableConfigs) > 0 else None

def importCells(file_path):
    cells = {}
    df = pd.read_csv(file_path)
    for row in df.itertuples(index=False, name=None):
        try:
            # Skip rows where capacity cannot be converted to float (likely headers)
            capacity = float(row[6])  # typical capacity (mAh)
            current = float(row[7])   # Peak discharge rating max (A)
            weight = float(row[11])   # weight (g)typical
            length = float(row[12])   # Length
            thickness = float(row[13]) # thickness (or diameter)
            width = float(row[14])    # width (if applicable)
            voltage = float(row[9])   # nominal voltage
        except (ValueError, IndexError):
            # Skip rows with non-numeric data or insufficient columns
            print(f"{str('#'*100)}\nSkipping row with invalid data: {row}\n{str('#'*100)}")
            continue
            
        # Vprint(f"Importing cell model: {row[3]}")  # Debug print
        # Vprint(f"Importing cell model: {row[3]}")
        cell = Cell(
            store=row[0],
            link=row[1],
            brand=row[3],     # brand
            model=row[4],     # model
            package=row[5],   # Package
            capacity_mAh=capacity,
            current_A=current,
            weight_g=weight,
            length_mm=length,
            thickness_mm=thickness,
            width_mm=width,
            voltage_V=voltage
        )
        cells[row[4]] = cell  # Use model as key instead of brand
    return cells

def GenerateConfigs(cells, MaxPowers=[70000, 80000, 90000], TargetVoltages=[300, 350, 400, 450, 510], PotentialModuleCounts=8):
    configs = {}
    total = len(cells) * len(MaxPowers) * len(TargetVoltages)
    with tqdm(total=total, desc="Generating configs") as pbar:
        for cell in cells.values():
            for MaxPower in MaxPowers:
                for TargetVoltage in TargetVoltages:
                    tempconfiglistCC = GenerateConfig_CC(cell, MaxPower, TargetVoltage, PotentialModuleCount=PotentialModuleCounts)
                    Vprint(type(tempconfiglistCC))
                    if tempconfiglistCC is not None and len(tempconfiglistCC) > 1:
                        for config in tempconfiglistCC:
                            configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{config.MCnt}M_CC"] = config
                    elif tempconfiglistCC is not None:
                        configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{tempconfiglistCC[0].MCnt}M_CC"] = tempconfiglistCC[0]
                    tempconfiglistCF = GenerateConfig_CF(cell, MaxPower, TargetVoltage, PotentialModuleCount=PotentialModuleCounts)
                    if tempconfiglistCF is not None and len(tempconfiglistCF) > 1:
                        for config in tempconfiglistCF:
                            configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{config.MCnt}M_CF"] = config
                    elif tempconfiglistCF is not None:
                        configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{tempconfiglistCF[0].MCnt}M_CF"] = tempconfiglistCF[0]
                    tempconfiglistFC = GenerateConfig_FC(cell, MaxPower, TargetVoltage, PotentialModuleCount=PotentialModuleCounts)
                    if tempconfiglistFC is not None and len(tempconfiglistFC) > 1:
                        for config in tempconfiglistFC:
                            configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{config.MCnt}M_FC"] = config
                    elif tempconfiglistFC is not None:
                        configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{tempconfiglistFC[0].MCnt}M_FC"] = tempconfiglistFC[0]
                    tempconfiglistFF = GenerateConfig_FF(cell, MaxPower, TargetVoltage, PotentialModuleCount=PotentialModuleCounts)
                    if tempconfiglistFF is not None and len(tempconfiglistFF) > 1:
                        for config in tempconfiglistFF:
                            configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{config.MCnt}M_FF"] = config
                    elif tempconfiglistFF is not None:
                        configs[f"{cell.model}_{MaxPower}kW_{TargetVoltage}V_{tempconfiglistFF[0].MCnt}M_FF"] = tempconfiglistFF[0]
                    pbar.update(1)
    return configs

def Configs_to_csv(configs, filename):
    Cell, SMod, P, MCnt, PackVoltage, CellWeight, S, Current, Energy, Power, TotalWallVolume, TotalWallWeight, PackLength, PackWidth, PackHeight, TotalPackWeight,TotalWallWeight, GravEnergyDensity, GravPowerDensity = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for config in configs.values():
        S.append(config.S)
        Cell.append(config.cell.model) #Cell Name
        SMod.append(config.SMod) #Number Cells in series in module
        P.append(config.P) #Number of Cells in Parallel
        MCnt.append(config.MCnt) #Number of Modules in Series
        PackVoltage.append(config.PackVoltage) #Pack Total Voltage
        CellWeight.append(config.CellWeight) #Weight of the pack attributable to the cells themselves
        Current.append(config.Current) #Pack Current
        TotalPackWeight.append(config.TotalPackWeight) #Pack Cumulative ESTIMATED Weight
        TotalWallWeight.append(config.TotalWallWeight) #Weight Attributable to the Walls
        Energy.append(config.Energy) #Pack Energy
        Power.append(config.Power) #Pack Power
        TotalWallVolume.append(config.TotalWallVolume) #Volume Attributable to the Walls (Mostly used as an intermediate but exported for checking purposes)
        GravEnergyDensity.append(config.GravEnergyDensity) #Energy/kg
        GravPowerDensity.append(config.GravPowerDensity) #Power/kg
        PackLength.append(config.PackLength) #Length of the pack
        PackWidth.append(config.PackWidth) #Width of the pack
        PackHeight.append(config.PackHeight) #Height of the pack
    pd.DataFrame({
        "Cell": Cell,
        "Pack Voltage (V)": PackVoltage,
        "SMod": SMod,
        "S": S,
        "Pack Current (A)": Current,
        "P": P,
        "MCnt": MCnt,
        "Pack Power (kW)": Power,
        "Pack Energy (kWh)": Energy,
        "Total Pack Weight (kg)": TotalPackWeight,
        "Cell Weight (kg)": CellWeight,
        "Total Wall Weight (kg)": TotalWallWeight,
        "Grav Energy Density (Wh/kg)": GravEnergyDensity,
        "Grav Power Density (W/kg)": GravPowerDensity,
        "Pack Length (mm)": PackLength,
        "Pack Width (mm)": PackWidth,
        "Pack Height (mm)": PackHeight,
        "Total Wall Volume (mm³)": TotalWallVolume

    }).to_csv(filename, index=False)

def CylOnly(cells):
    for i in list(cells.keys()):
        if cells[i].package != "Cylinder":
            cells.pop(i)
    return cells

def GenerateConfigs_Exhaustive(cells, MaxPower=140000, ModuleMaxVoltage=100, ModuleMaxE=6000000, ModuleMaxCellWeight=8000, PotentialModuleCounts=8):
    configs = {}
    total = len(cells)
    with tqdm(total=total, desc="Generating exhaustive configs") as pbar:
        for cell in cells.values():
            tempconfigs = GenerateConfig_Exhaustive(
                cell,
                MaxPower=MaxPower,
                ModuleMaxVoltage=ModuleMaxVoltage,
                ModuleMaxE=ModuleMaxE,
                ModuleMaxCellWeight=ModuleMaxCellWeight,
                PotentialModuleCount=PotentialModuleCounts
            )
            if tempconfigs is not None:
                for i, config in enumerate(tempconfigs):
                    configs[f"{cell.model}_exh_{i}"] = config
            pbar.update(1)
    return configs

if __name__ == "__main__":
    if not os.path.exists(CellsFilePath):
        print("cells.csv not found")
    else:
        print("Importing all cells from CSV")
        all_cells = importCells(CellsFilePath)
        cyl_cells = CylOnly(all_cells.copy())

        print(f"Imported {len(cyl_cells)} cylindrical cells and {len(all_cells)} total cells")

        if USE_EXHAUSTIVE:
            # Generate configs for cylindrical cells (exhaustive)
            configs_cyl = GenerateConfigs_Exhaustive(cyl_cells, MaxPower=80000)
            Configs_to_csv(configs_cyl, "ConfigCyl.csv")
            print(f"Saved {len(configs_cyl)} cylindrical configs to ConfigCyl.csv")

            # Generate configs for all cells (exhaustive)
            configs_all = GenerateConfigs_Exhaustive(all_cells, MaxPower=80000)
            Configs_to_csv(configs_all, "ConfigAll.csv")
            print(f"Saved {len(configs_all)} all-cell configs to ConfigAll.csv")
        else:
            # Generate configs for cylindrical cells (original)
            configs_cyl = GenerateConfigs(cyl_cells)
            Configs_to_csv(configs_cyl, "ConfigCyl.csv")
            print(f"Saved {len(configs_cyl)} cylindrical configs to ConfigCyl.csv")

            # Generate configs for all cells (original)
            configs_all = GenerateConfigs(all_cells)
            Configs_to_csv(configs_all, "ConfigAll.csv")
            print(f"Saved {len(configs_all)} all-cell configs to ConfigAll.csv")