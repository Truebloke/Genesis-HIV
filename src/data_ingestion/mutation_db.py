"""
SQL database module for storing HIV mutation and drug resistance data.

This module handles the creation and management of a SQLite database
to store information about HIV mutations, drug resistance patterns,
and associated metadata.
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Any
import os


class MutationDatabase:
    """
    A class to manage the HIV mutation and drug resistance database.
    
    This includes:
    - Creating database tables
    - Storing mutation data
    - Querying resistance patterns
    - Managing drug information
    """
    
    def __init__(self, db_path: str = "./data/mutations.db"):
        """
        Initialize the MutationDatabase.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize the database
        self.init_database()
    
    def init_database(self) -> None:
        """Initialize the database with required tables."""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        # Create table for drugs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drugs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                class TEXT NOT NULL,
                mechanism TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create table for mutations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mutations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gene TEXT NOT NULL,
                mutation_code TEXT NOT NULL,
                position INTEGER NOT NULL,
                wildtype_aa TEXT NOT NULL,
                mutant_aa TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(gene, position, wildtype_aa, mutant_aa)
            )
        """)
        
        # Create table for drug resistance associations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_resistance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mutation_id INTEGER NOT NULL,
                drug_id INTEGER NOT NULL,
                fold_change REAL,
                phenotype TEXT,
                level TEXT,
                reference TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (mutation_id) REFERENCES mutations(id),
                FOREIGN KEY (drug_id) REFERENCES drugs(id),
                UNIQUE(mutation_id, drug_id)
            )
        """)
        
        # Create table for patient sequences
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                sequence TEXT NOT NULL,
                date_sampled DATE,
                viral_load REAL,
                cd4_count INTEGER,
                treatment_history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create table for patient mutations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_mutations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_seq_id INTEGER NOT NULL,
                mutation_id INTEGER NOT NULL,
                frequency REAL,
                coverage INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_seq_id) REFERENCES patient_sequences(id),
                FOREIGN KEY (mutation_id) REFERENCES mutations(id)
            )
        """)
        
        self.connection.commit()
        print(f"Database initialized at {self.db_path}")
    
    def add_drug(self, name: str, drug_class: str, mechanism: str = None) -> int:
        """
        Add a drug to the database.
        
        Args:
            name (str): Name of the drug
            drug_class (str): Class of the drug (e.g., NNRTI, PI, NRTI)
            mechanism (str, optional): Mechanism of action
            
        Returns:
            int: ID of the added drug
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO drugs (name, class, mechanism) VALUES (?, ?, ?)",
                (name, drug_class, mechanism)
            )
            self.connection.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # If drug already exists, return its ID
            cursor.execute("SELECT id FROM drugs WHERE name = ?", (name,))
            return cursor.fetchone()[0]
    
    def add_mutation(self, gene: str, mutation_code: str, position: int, 
                     wildtype_aa: str, mutant_aa: str, description: str = None) -> int:
        """
        Add a mutation to the database.
        
        Args:
            gene (str): Gene name (e.g., PR, RT, IN)
            mutation_code (str): Mutation code (e.g., K103N, M184V)
            position (int): Position in the gene
            wildtype_aa (str): Wild-type amino acid
            mutant_aa (str): Mutant amino acid
            description (str, optional): Description of the mutation
            
        Returns:
            int: ID of the added mutation
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO mutations (gene, mutation_code, position, wildtype_aa, mutant_aa, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (gene, mutation_code, position, wildtype_aa, mutant_aa, description))
            self.connection.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # If mutation already exists, return its ID
            cursor.execute(
                "SELECT id FROM mutations WHERE gene = ? AND position = ? AND wildtype_aa = ? AND mutant_aa = ?",
                (gene, position, wildtype_aa, mutant_aa)
            )
            return cursor.fetchone()[0]
    
    def add_drug_resistance(self, mutation_id: int, drug_id: int, fold_change: float = None,
                           phenotype: str = None, level: str = None, reference: str = None) -> int:
        """
        Associate a mutation with drug resistance.
        
        Args:
            mutation_id (int): ID of the mutation
            drug_id (int): ID of the drug
            fold_change (float, optional): Fold change in IC50
            phenotype (str, optional): Resistance phenotype
            level (str, optional): Resistance level (e.g., low, medium, high)
            reference (str, optional): Reference for the data
            
        Returns:
            int: ID of the added resistance association
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO drug_resistance (mutation_id, drug_id, fold_change, phenotype, level, reference)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mutation_id, drug_id, fold_change, phenotype, level, reference))
            self.connection.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # If association already exists, return its ID
            cursor.execute(
                "SELECT id FROM drug_resistance WHERE mutation_id = ? AND drug_id = ?",
                (mutation_id, drug_id)
            )
            return cursor.fetchone()[0]
    
    def add_patient_sequence(self, patient_id: str, sequence: str, date_sampled: str = None,
                            viral_load: float = None, cd4_count: int = None, 
                            treatment_history: str = None) -> int:
        """
        Add a patient sequence to the database.
        
        Args:
            patient_id (str): Unique identifier for the patient
            sequence (str): HIV sequence
            date_sampled (str, optional): Date when sample was taken
            viral_load (float, optional): Viral load measurement
            cd4_count (int, optional): CD4 count
            treatment_history (str, optional): Treatment history
            
        Returns:
            int: ID of the added patient sequence
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO patient_sequences (patient_id, sequence, date_sampled, viral_load, cd4_count, treatment_history)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_id, sequence, date_sampled, viral_load, cd4_count, treatment_history))
        self.connection.commit()
        return cursor.lastrowid
    
    def add_patient_mutation(self, patient_seq_id: int, mutation_id: int, 
                             frequency: float = None, coverage: int = None) -> int:
        """
        Associate a mutation with a patient sequence.
        
        Args:
            patient_seq_id (int): ID of the patient sequence
            mutation_id (int): ID of the mutation
            frequency (float, optional): Frequency of the mutation in the sample
            coverage (int, optional): Coverage depth for the mutation
            
        Returns:
            int: ID of the added patient mutation association
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO patient_mutations (patient_seq_id, mutation_id, frequency, coverage)
            VALUES (?, ?, ?, ?)
        """, (patient_seq_id, mutation_id, frequency, coverage))
        self.connection.commit()
        return cursor.lastrowid
    
    def get_resistance_profile(self, drug_name: str) -> pd.DataFrame:
        """
        Get all mutations associated with resistance to a specific drug.
        
        Args:
            drug_name (str): Name of the drug
            
        Returns:
            pd.DataFrame: DataFrame with mutation and resistance information
        """
        query = """
            SELECT m.mutation_code, m.gene, dr.fold_change, dr.phenotype, dr.level
            FROM drug_resistance dr
            JOIN mutations m ON dr.mutation_id = m.id
            JOIN drugs d ON dr.drug_id = d.id
            WHERE d.name = ?
            ORDER BY dr.fold_change DESC
        """
        df = pd.read_sql_query(query, self.connection, params=(drug_name,))
        return df
    
    def get_drug_resistance_mutations(self, drug_class: str = None) -> pd.DataFrame:
        """
        Get all mutations associated with drug resistance.
        
        Args:
            drug_class (str, optional): Drug class to filter by
            
        Returns:
            pd.DataFrame: DataFrame with mutation and resistance information
        """
        if drug_class:
            query = """
                SELECT d.name AS drug, d.class AS drug_class, m.mutation_code, m.gene, 
                       dr.fold_change, dr.phenotype, dr.level
                FROM drug_resistance dr
                JOIN mutations m ON dr.mutation_id = m.id
                JOIN drugs d ON dr.drug_id = d.id
                WHERE d.class = ?
                ORDER BY d.name, dr.fold_change DESC
            """
            params = (drug_class,)
        else:
            query = """
                SELECT d.name AS drug, d.class AS drug_class, m.mutation_code, m.gene, 
                       dr.fold_change, dr.phenotype, dr.level
                FROM drug_resistance dr
                JOIN mutations m ON dr.mutation_id = m.id
                JOIN drugs d ON dr.drug_id = d.id
                ORDER BY d.class, d.name, dr.fold_change DESC
            """
            params = ()
        
        df = pd.read_sql_query(query, self.connection, params=params)
        return df
    
    def predict_resistance(self, mutation_list: List[str]) -> Dict[str, Any]:
        """
        Predict drug resistance based on a list of mutations.
        
        Args:
            mutation_list (List[str]): List of mutation codes (e.g., ['K103N', 'M184V'])
            
        Returns:
            Dict[str, Any]: Predicted resistance profile
        """
        # Get mutation IDs
        placeholders = ','.join(['?' for _ in mutation_list])
        query = f"""
            SELECT DISTINCT dr.drug_id
            FROM drug_resistance dr
            JOIN mutations m ON dr.mutation_id = m.id
            WHERE m.mutation_code IN ({placeholders})
        """
        
        cursor = self.connection.cursor()
        cursor.execute(query, mutation_list)
        drug_ids = [row[0] for row in cursor.fetchall()]
        
        # Get drug names and resistance levels
        if drug_ids:
            placeholders = ','.join(['?' for _ in drug_ids])
            query = f"""
                SELECT d.name, d.class, dr.level, dr.fold_change, m.mutation_code
                FROM drug_resistance dr
                JOIN drugs d ON dr.drug_id = d.id
                JOIN mutations m ON dr.mutation_id = m.id
                WHERE dr.drug_id IN ({placeholders})
                ORDER BY d.class, d.name
            """
            cursor.execute(query, drug_ids)
            
            results = {}
            for drug_name, drug_class, level, fold_change, mutation_code in cursor.fetchall():
                if drug_name not in results:
                    results[drug_name] = {
                        'class': drug_class,
                        'level': level,
                        'fold_change': fold_change,
                        'mutations': []
                    }
                
                if mutation_code not in results[drug_name]['mutations']:
                    results[drug_name]['mutations'].append(mutation_code)
            
            return results
        else:
            return {}
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    # Example usage
    db = MutationDatabase()
    
    # Add some drugs
    efv_id = db.add_drug("Efavirenz", "NNRTI", "Non-nucleoside reverse transcriptase inhibitor")
    tdf_id = db.add_drug("Tenofovir", "NRTI", "Nucleoside reverse transcriptase inhibitor")
    lmv_id = db.add_drug("Lamivudine", "NRTI", "Nucleoside reverse transcriptase inhibitor")
    
    # Add some mutations
    k103n_id = db.add_mutation("RT", "K103N", 103, "K", "N", "Common NNRTI resistance mutation")
    m184v_id = db.add_mutation("RT", "M184V", 184, "M", "V", "NRTI resistance mutation affecting fitness")
    
    # Associate mutations with drugs
    db.add_drug_resistance(k103n_id, efv_id, fold_change=100.0, 
                          phenotype="High-level resistance", level="High")
    db.add_drug_resistance(m184v_id, lmv_id, fold_change=0.1, 
                          phenotype="High-level resistance", level="High")
    db.add_drug_resistance(m184v_id, tdf_id, fold_change=2.5, 
                          phenotype="Low-level resistance", level="Low")
    
    # Get resistance profile for a drug
    efv_resistances = db.get_resistance_profile("Efavirenz")
    print("Efavirenz resistance mutations:")
    print(efv_resistances)
    
    # Predict resistance for a mutation list
    mutation_list = ["K103N", "M184V"]
    predicted_resistance = db.predict_resistance(mutation_list)
    print("\nPredicted resistance for mutations:", mutation_list)
    for drug, info in predicted_resistance.items():
        print(f"  {drug} ({info['class']}): {info['level']} resistance due to {info['mutations']}")
    
    # Close the database
    db.close()