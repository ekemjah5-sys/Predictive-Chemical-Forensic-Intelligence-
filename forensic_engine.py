import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

print("--- Phase 1: Initializing Forensic Molecular Repository ---")

# We define our dual-route system. 
# This dictionary holds the 'Chemical Fingerprint Clues' for both Nigerian threat tracks.
molecular_repository = {
    # TRACK 1: THE INDUSTRIAL P2P LEUCKART ROUTE
    "P2P_Precursor": {
        "smiles": "CC(=O)CC1=CC=CC=C1", 
        "route": "P2P_Leuckart", 
        "type": "Precursor"
    },
    "N_Formylmethamphetamine": {
        "smiles": "CC(CC1=CC=CC=C1)N(C)C=O", 
        "route": "P2P_Leuckart", 
        "type": "Key_Route_Impurity"
    },
    "Dibenzyl_Ketone_Impurity": {
        "smiles": "C1=CC=C(C=C1)CC(=O)CC2=CC=CC=C2", 
        "route": "P2P_Leuckart", 
        "type": "Side_Reaction_Marker"
    },
    
    # TRACK 2: THE MOBILE EPHEDRINE-IODINE ROUTE
    "Ephedrine_Precursor": {
        "smiles": "CC(C(C1=CC=CC=C1)O)NC", 
        "route": "Ephedrine_Iodine", 
        "type": "Precursor"
    },
    "Cis_Aziridine_Impurity": {
        "smiles": "CC1N(C)C1C2=CC=CC=C2", 
        "route": "Ephedrine_Iodine", 
        "type": "Key_Route_Impurity"
    },
    "Chloroephedrine_Intermediate": {
        "smiles": "CC(C(C1=CC=CC=C1)Cl)NC", 
        "route": "Ephedrine_Iodine", 
        "type": "Incomplete_Reaction_Marker"
    },
    
    # THE CRIME SCENE TARGET OUTPUT
    "Methamphetamine_Product": {
        "smiles": "CC(CC1=CC=CC=C1)NC", 
        "route": "Universal", 
        "type": "Seized_Target_Drug"
    }
}

# Create an empty list to capture our calculated database rows
database_records = []

print("\n--- Phase 2: Processing 3D Structure Optimization Engine ---")

for name, data in molecular_repository.items():
    # 1. Parse the text string into a formal chemical object
    mol = Chem.MolFromSmiles(data["smiles"])
    
    # 2. Add structural hydrogens required for accurate mass and geometry modeling
    mol = Chem.AddHs(mol)
    
    # 3. Project the flat molecule into a 3D coordinate system using ETKDGv3 algorithm
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    
    # 4. Apply the Merck Molecular Force Field (MMFF94) to optimize the structure.
    # It returns '0' if the structural geometry successfully reaches an equilibrium state.
    optimization_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    
    # 5. Extract calculated descriptors to serve as features for our future Machine Learning engine
    mol_weight = Descriptors.MolWt(mol)    # Total molecular mass
    log_p = Descriptors.MolLogP(mol)        # Solublity/partition coefficient marker
    tpsa = Descriptors.TPSA(mol)            # Total Polar Surface Area (molecular polarity)
    
    # Append the calculated real-world data points into our framework memory
    database_records.append({
        "Molecule_Name": name,
        "Route_Origin": data["route"],
        "Forensic_Role": data["type"],
        "Mol_Weight_g_mol": round(mol_weight, 3),
        "LogP_Polarity": round(log_p, 3),
        "TPSA_Surface": round(tpsa, 3),
        "Geometry_Optimized": "SUCCESS" if optimization_status == 0 else "FAILED"
    })
    
    print(f" -> Successfully mapped, hydrogenated, and 3D-optimized: {name}")

# Phase 3: Display the finished database matrix
print("\n--- Phase 3: Final Forensic Intelligence Database Generated ---")
forensic_df = pd.DataFrame(database_records)

# Configure pandas options to print out a highly legible, clean text table in your prompt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(forensic_df.to_string(index=False))
