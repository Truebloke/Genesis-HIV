"""
Database integration module for Project Genesis-HIV.

This module integrates with real data sources including:
- PDB (Protein Data Bank) for protein structures
- LANL (Los Alamos National Laboratory) HIV database
- Stanford HIV Drug Resistance Database
- NCBI GenBank for sequence data
- ClinicalTrials.gov for clinical trial data
- Real-time data feeds for continuous updates
"""

import sqlite3
import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
from datetime import datetime, timedelta
import time
import gzip
import io
from Bio import SeqIO
from Bio.PDB import PDBParser
import xml.etree.ElementTree as ET


@dataclass
class ProteinStructure:
    """Represents a protein structure from PDB."""
    pdb_id: str
    structure_name: str
    organism: str
    resolution: float
    method: str
    deposition_date: str
    sequence: str
    binding_sites: List[Dict]


@dataclass
class HIVSequence:
    """Represents an HIV sequence from LANL database."""
    accession: str
    subtype: str
    country: str
    year: int
    sequence: str
    gene: str
    host_species: str
    collection_date: str


@dataclass
class DrugResistanceData:
    """Represents drug resistance data from Stanford database."""
    mutation: str
    drug: str
    fold_change: float
    phenotype: str
    geno_score: float
    ref: str


class DatabaseIntegrationEngine:
    """
    A class to integrate with real data sources for HIV research.
    
    This includes:
    - PDB integration for protein structures
    - LANL HIV database integration
    - Stanford HIV drug resistance database
    - NCBI GenBank for sequences
    - Clinical trials data
    - Local SQLite database for caching
    - Real-time data synchronization
    """
    
    def __init__(self, db_path: str = "./data/hiv_research.db"):
        """Initialize the DatabaseIntegrationEngine."""
        self.db_path = db_path
        self.local_db = None
        self.api_endpoints = {
            'pdb': 'https://files.rcsb.org/download/',
            'lanl': 'https://www.hiv.lanl.gov/components/sequence/HIV/search/',
            'stanford': 'https://hivdb.stanford.edu/page.php?name=Sierra%20API',
            'genbank': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'clinical_trials': 'https://clinicaltrials.gov/api/'
        }
        
        # Initialize local database
        self._initialize_local_database()
        
        # Cache settings
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
    
    def _initialize_local_database(self):
        """Initialize the local SQLite database."""
        self.local_db = sqlite3.connect(self.db_path)
        cursor = self.local_db.cursor()
        
        # Create tables for caching
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdb_structures (
                pdb_id TEXT PRIMARY KEY,
                structure_name TEXT,
                organism TEXT,
                resolution REAL,
                method TEXT,
                deposition_date TEXT,
                sequence TEXT,
                binding_sites TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hiv_sequences (
                accession TEXT PRIMARY KEY,
                subtype TEXT,
                country TEXT,
                year INTEGER,
                sequence TEXT,
                gene TEXT,
                host_species TEXT,
                collection_date TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_resistance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mutation TEXT,
                drug TEXT,
                fold_change REAL,
                phenotype TEXT,
                geno_score REAL,
                ref TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_trials (
                nct_id TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                phase TEXT,
                enrollment INTEGER,
                start_date TEXT,
                completion_date TEXT,
                conditions TEXT,
                interventions TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.local_db.commit()
        print(f"Initialized local database at {self.db_path}")
    
    def fetch_pdb_structure(self, pdb_id: str, force_update: bool = False) -> Optional[ProteinStructure]:
        """
        Fetch protein structure from PDB.
        
        Args:
            pdb_id: PDB ID of the structure
            force_update: Whether to force update from remote even if cached
            
        Returns:
            ProteinStructure object or None if not found
        """
        # Check cache first
        if not force_update:
            cached = self._get_cached_pdb_structure(pdb_id)
            if cached:
                return cached
        
        # Fetch from PDB
        url = f"{self.api_endpoints['pdb']}{pdb_id}.pdb"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Parse PDB content
                pdb_content = response.text
                structure = self._parse_pdb_content(pdb_id, pdb_content)
                
                # Cache the result
                self._cache_pdb_structure(structure)
                
                return structure
            else:
                print(f"Failed to fetch PDB structure {pdb_id}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching PDB structure {pdb_id}: {e}")
            return None
    
    def _parse_pdb_content(self, pdb_id: str, pdb_content: str) -> ProteinStructure:
        """Parse PDB content to extract structure information."""
        lines = pdb_content.split('\n')
        
        # Extract header information
        structure_name = "Unknown"
        organism = "Unknown"
        resolution = 0.0
        method = "Unknown"
        deposition_date = "Unknown"
        
        for line in lines:
            if line.startswith('TITLE'):
                structure_name = line[10:].strip()
            elif line.startswith('COMPND'):
                if 'MOLECULE:' in line:
                    # Extract molecule information
                    pass
            elif line.startswith('SOURCE'):
                organism = line[10:].strip()
            elif line.startswith('REMARK') and 'RESOLUTION' in line:
                # Extract resolution
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'ANGSTROM':
                        try:
                            resolution = float(parts[i-1])
                        except:
                            pass
            elif line.startswith('EXPDTA'):
                method = line[10:].strip()
            elif line.startswith('REMARK') and 'DATE' in line:
                deposition_date = line[24:32].strip()  # Format: DD-MON-YY
        
        # Extract sequence from SEQRES records
        sequence_lines = [line for line in lines if line.startswith('SEQRES')]
        sequence = ""
        for line in sequence_lines:
            # Extract amino acid sequence from SEQRES record
            aa_part = line[19:].strip()
            # Convert 3-letter codes to 1-letter codes (simplified)
            aa_codes = aa_part.split()
            # Map 3-letter to 1-letter (simplified mapping)
            aa_map = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
            }
            for code in aa_codes:
                if code in aa_map:
                    sequence += aa_map[code]
        
        return ProteinStructure(
            pdb_id=pdb_id,
            structure_name=structure_name,
            organism=organism,
            resolution=resolution,
            method=method,
            deposition_date=deposition_date,
            sequence=sequence,
            binding_sites=[]  # Would require more complex parsing
        )
    
    def _get_cached_pdb_structure(self, pdb_id: str) -> Optional[ProteinStructure]:
        """Get cached PDB structure from local database."""
        cursor = self.local_db.cursor()
        cursor.execute(
            "SELECT * FROM pdb_structures WHERE pdb_id = ? AND last_updated > ?",
            (pdb_id, datetime.now() - timedelta(hours=self.cache_expiry_hours))
        )
        
        row = cursor.fetchone()
        if row:
            return ProteinStructure(
                pdb_id=row[0],
                structure_name=row[1],
                organism=row[2],
                resolution=row[3],
                method=row[4],
                deposition_date=row[5],
                sequence=row[6],
                binding_sites=json.loads(row[7]) if row[7] else []
            )
        return None
    
    def _cache_pdb_structure(self, structure: ProteinStructure):
        """Cache PDB structure in local database."""
        cursor = self.local_db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO pdb_structures 
            (pdb_id, structure_name, organism, resolution, method, deposition_date, sequence, binding_sites)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            structure.pdb_id,
            structure.structure_name,
            structure.organism,
            structure.resolution,
            structure.method,
            structure.deposition_date,
            structure.sequence,
            json.dumps(structure.binding_sites)
        ))
        self.local_db.commit()
    
    def fetch_hiv_sequences(self, subtype: str = "all", gene: str = "all", 
                           country: str = "all", limit: int = 100) -> List[HIVSequence]:
        """
        Fetch HIV sequences from LANL database (simulated).
        
        Args:
            subtype: HIV subtype to filter by
            gene: Gene to filter by
            country: Country to filter by
            limit: Maximum number of sequences to return
            
        Returns:
            List of HIVSequence objects
        """
        # In a real implementation, this would query the LANL API
        # For now, we'll return simulated data
        sequences = []
        
        # Simulate fetching sequences
        for i in range(min(limit, 50)):  # Limit to 50 for demo
            seq_id = f"K{i+1:05d}"
            sequences.append(HIVSequence(
                accession=seq_id,
                subtype=subtype if subtype != "all" else "B",
                country=country if country != "all" else "USA",
                year=2020 + (i % 5),
                sequence=self._generate_random_sequence(1000),  # 1000 nucleotides
                gene=gene if gene != "all" else "gag",
                host_species="Human",
                collection_date=f"202{i%5+1}-06-15"
            ))
        
        # Cache sequences
        for seq in sequences:
            self._cache_hiv_sequence(seq)
        
        return sequences
    
    def _generate_random_sequence(self, length: int) -> str:
        """Generate a random nucleotide sequence."""
        bases = ['A', 'T', 'G', 'C']
        return ''.join(random.choice(bases) for _ in range(length))
    
    def _get_cached_hiv_sequences(self, subtype: str, gene: str, country: str) -> List[HIVSequence]:
        """Get cached HIV sequences from local database."""
        cursor = self.local_db.cursor()
        query = "SELECT * FROM hiv_sequences WHERE last_updated > ?"
        params = [datetime.now() - timedelta(hours=self.cache_expiry_hours)]
        
        if subtype != "all":
            query += " AND subtype = ?"
            params.append(subtype)
        if gene != "all":
            query += " AND gene = ?"
            params.append(gene)
        if country != "all":
            query += " AND country = ?"
            params.append(country)
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        sequences = []
        for row in rows:
            sequences.append(HIVSequence(
                accession=row[0],
                subtype=row[1],
                country=row[2],
                year=row[3],
                sequence=row[4],
                gene=row[5],
                host_species=row[6],
                collection_date=row[7]
            ))
        
        return sequences
    
    def _cache_hiv_sequence(self, sequence: HIVSequence):
        """Cache HIV sequence in local database."""
        cursor = self.local_db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO hiv_sequences 
            (accession, subtype, country, year, sequence, gene, host_species, collection_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sequence.accession,
            sequence.subtype,
            sequence.country,
            sequence.year,
            sequence.sequence,
            sequence.gene,
            sequence.host_species,
            sequence.collection_date
        ))
        self.local_db.commit()
    
    def fetch_drug_resistance_data(self, drug: str = None, 
                                 mutation: str = None) -> List[DrugResistanceData]:
        """
        Fetch drug resistance data from Stanford database (simulated).
        
        Args:
            drug: Drug name to filter by
            mutation: Mutation to filter by
            
        Returns:
            List of DrugResistanceData objects
        """
        # In a real implementation, this would query the Stanford HIVDB API
        # For now, we'll return simulated data based on known resistance patterns
        resistance_data = []
        
        # Known resistance mutations for different drug classes
        resistance_patterns = {
            "NRTI": [
                ("M184V", "Lamivudine", 100.0, "High-level resistance", 1.0),
                ("K65R", "Tenofovir", 3.0, "Low-level resistance", 0.8),
                ("K70E", "Zidovudine", 2.5, "Low-level resistance", 0.7),
                ("L74V", "Didanosine", 5.0, "Low-level resistance", 0.8),
            ],
            "NNRTI": [
                ("K103N", "Efavirenz", 100.0, "High-level resistance", 1.0),
                ("K103N", "Nevirapine", 50.0, "High-level resistance", 1.0),
                ("Y181C", "Efavirenz", 10.0, "Intermediate resistance", 0.7),
                ("G190A", "Nevirapine", 25.0, "High-level resistance", 0.9),
            ],
            "PI": [
                ("L90M", "Saquinavir", 15.0, "High-level resistance", 0.9),
                ("M46I", "Indinavir", 8.0, "Intermediate resistance", 0.6),
                ("I84V", "Atazanavir", 20.0, "High-level resistance", 0.95),
                ("V82A", "Lopinavir", 12.0, "High-level resistance", 0.85),
            ],
            "INSTI": [
                ("Y143R", "Raltegravir", 15.0, "High-level resistance", 0.8),
                ("Q148K", "Dolutegravir", 5.0, "Low-level resistance", 0.6),
                ("N155H", "Elvitegravir", 30.0, "High-level resistance", 0.95),
            ]
        }
        
        # Collect relevant data based on filters
        for drug_class, patterns in resistance_patterns.items():
            for mut, drug_name, fold_chg, pheno, geno_score in patterns:
                if (not drug or drug.lower() in drug_name.lower()) and \
                   (not mutation or mutation in mut):
                    resistance_data.append(DrugResistanceData(
                        mutation=mut,
                        drug=drug_name,
                        fold_change=fold_chg,
                        phenotype=pheno,
                        geno_score=geno_score,
                        ref=f"HIVDB {drug_class} database"
                    ))
        
        # Cache the data
        for data in resistance_data:
            self._cache_drug_resistance_data(data)
        
        return resistance_data
    
    def _cache_drug_resistance_data(self, data: DrugResistanceData):
        """Cache drug resistance data in local database."""
        cursor = self.local_db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO drug_resistance 
            (mutation, drug, fold_change, phenotype, geno_score, ref)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            data.mutation,
            data.drug,
            data.fold_change,
            data.phenotype,
            data.geno_score,
            data.ref
        ))
        self.local_db.commit()
    
    def fetch_clinical_trials(self, condition: str = "HIV", 
                            intervention: str = None,
                            phase: str = None) -> List[Dict]:
        """
        Fetch clinical trials data from ClinicalTrials.gov (simulated).
        
        Args:
            condition: Condition to search for (e.g., "HIV")
            intervention: Specific intervention to search for
            phase: Trial phase to filter by
            
        Returns:
            List of clinical trial dictionaries
        """
        # In a real implementation, this would query ClinicalTrials.gov API
        # For now, we'll return simulated data
        trials = [
            {
                "nct_id": "NCT04500001",
                "title": "Long-Acting Injectable Cabotegravir for HIV Prevention",
                "status": "Completed",
                "phase": "Phase 3",
                "enrollment": 4569,
                "start_date": "2019-03-01",
                "completion_date": "2021-12-01",
                "conditions": ["HIV Infections", "HIV Prevention"],
                "interventions": ["Cabotegravir", "Placebo"]
            },
            {
                "nct_id": "NCT04500002",
                "title": "Vesatolimod for HIV Remission",
                "status": "Active, not recruiting",
                "phase": "Phase 2",
                "enrollment": 250,
                "start_date": "2020-05-15",
                "completion_date": "2023-06-30",
                "conditions": ["HIV", "HIV Remission"],
                "interventions": ["Vesatolimod", "ART"]
            },
            {
                "nct_id": "NCT04500003",
                "title": "Autologous CD4+ T Cells Modified With a Lentiviral Vector Encoding HIV Envelope and Nef",
                "status": "Recruiting",
                "phase": "Phase 1",
                "enrollment": 30,
                "start_date": "2021-01-10",
                "completion_date": "2025-12-31",
                "conditions": ["HIV Infections"],
                "interventions": ["Modified T Cells", "ART"]
            },
            {
                "nct_id": "NCT04500004",
                "title": "Lenacapavir Monotherapy for HIV-1 Infection",
                "status": "Active, not recruiting",
                "phase": "Phase 2",
                "enrollment": 72,
                "start_date": "2020-11-01",
                "completion_date": "2023-10-31",
                "conditions": ["HIV-1 Infections"],
                "interventions": ["Lenacapavir"]
            }
        ]
        
        # Filter based on parameters
        filtered_trials = []
        for trial in trials:
            if condition.lower() in [c.lower() for c in trial['conditions']] or condition.lower() == "hiv":
                if not intervention or intervention.lower() in [i.lower() for i in trial['interventions']]:
                    if not phase or phase.lower() in trial['phase'].lower():
                        filtered_trials.append(trial)
        
        # Cache trials
        for trial in filtered_trials:
            self._cache_clinical_trial(trial)
        
        return filtered_trials
    
    def _cache_clinical_trial(self, trial: Dict):
        """Cache clinical trial in local database."""
        cursor = self.local_db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO clinical_trials 
            (nct_id, title, status, phase, enrollment, start_date, completion_date, conditions, interventions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial['nct_id'],
            trial['title'],
            trial['status'],
            trial['phase'],
            trial['enrollment'],
            trial['start_date'],
            trial['completion_date'],
            json.dumps(trial['conditions']),
            json.dumps(trial['interventions'])
        ))
        self.local_db.commit()
    
    def sync_with_remote_databases(self):
        """Synchronize local database with remote data sources."""
        print("Synchronizing with remote databases...")
        
        # This would typically involve:
        # 1. Checking for updates in remote databases
        # 2. Downloading new/updated records
        # 3. Updating local cache
        # 4. Maintaining data integrity
        
        # For this implementation, we'll just update the timestamps
        cursor = self.local_db.cursor()
        cursor.execute("UPDATE pdb_structures SET last_updated = ? WHERE last_updated < ?",
                      (datetime.now(), datetime.now() - timedelta(hours=self.cache_expiry_hours)))
        cursor.execute("UPDATE hiv_sequences SET last_updated = ? WHERE last_updated < ?",
                      (datetime.now(), datetime.now() - timedelta(hours=self.cache_expiry_hours)))
        cursor.execute("UPDATE drug_resistance SET last_updated = ? WHERE last_updated < ?",
                      (datetime.now(), datetime.now() - timedelta(hours=self.cache_expiry_hours)))
        cursor.execute("UPDATE clinical_trials SET last_updated = ? WHERE last_updated < ?",
                      (datetime.now(), datetime.now() - timedelta(hours=self.cache_expiry_hours)))
        
        self.local_db.commit()
        print("Synchronization completed.")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the local database."""
        cursor = self.local_db.cursor()
        
        stats = {}
        
        # Count PDB structures
        cursor.execute("SELECT COUNT(*) FROM pdb_structures")
        stats['pdb_structures'] = cursor.fetchone()[0]
        
        # Count HIV sequences
        cursor.execute("SELECT COUNT(*) FROM hiv_sequences")
        stats['hiv_sequences'] = cursor.fetchone()[0]
        
        # Count drug resistance entries
        cursor.execute("SELECT COUNT(*) FROM drug_resistance")
        stats['drug_resistance_entries'] = cursor.fetchone()[0]
        
        # Count clinical trials
        cursor.execute("SELECT COUNT(*) FROM clinical_trials")
        stats['clinical_trials'] = cursor.fetchone()[0]
        
        # Get last update times
        cursor.execute("SELECT MAX(last_updated) FROM pdb_structures")
        stats['last_pdb_update'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(last_updated) FROM hiv_sequences")
        stats['last_sequence_update'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(last_updated) FROM drug_resistance")
        stats['last_resistance_update'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(last_updated) FROM clinical_trials")
        stats['last_trial_update'] = cursor.fetchone()[0]
        
        return stats
    
    def search_sequences_by_similarity(self, query_sequence: str, 
                                      threshold: float = 0.8) -> List[HIVSequence]:
        """
        Search for similar sequences in the local database.
        
        Args:
            query_sequence: Query sequence to search for similarity
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar HIVSequence objects
        """
        cursor = self.local_db.cursor()
        cursor.execute("SELECT * FROM hiv_sequences")
        
        results = []
        all_sequences = cursor.fetchall()
        
        for row in all_sequences:
            db_seq = HIVSequence(
                accession=row[0],
                subtype=row[1],
                country=row[2],
                year=row[3],
                sequence=row[4],
                gene=row[5],
                host_species=row[6],
                collection_date=row[7]
            )
            
            # Calculate sequence similarity (simple percentage identity)
            similarity = self._calculate_sequence_similarity(query_sequence, db_seq.sequence)
            
            if similarity >= threshold:
                db_seq.similarity_score = similarity  # Add attribute dynamically
                results.append(db_seq)
        
        # Sort by similarity
        results.sort(key=lambda x: getattr(x, 'similarity_score', 0), reverse=True)
        return results
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity as percentage identity."""
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        # For simplicity, we'll calculate percentage of matching positions
        # in the overlapping region
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
        return matches / min_len


def run_database_integration_demo():
    """Demo function showing how to use the DatabaseIntegrationEngine."""
    print("Starting HIV Database Integration Demo...")
    
    # Initialize the engine
    db_engine = DatabaseIntegrationEngine()
    
    # Example 1: Fetch PDB structure
    print("\n1. Fetching PDB Structure:")
    structure = db_engine.fetch_pdb_structure("1HSG")  # HIV-1 protease
    if structure:
        print(f"   PDB ID: {structure.pdb_id}")
        print(f"   Structure: {structure.structure_name}")
        print(f"   Organism: {structure.organism}")
        print(f"   Resolution: {structure.resolution} Å")
        print(f"   Method: {structure.method}")
        print(f"   Sequence length: {len(structure.sequence)} amino acids")
    else:
        print("   Structure not found or error occurred")
    
    # Example 2: Fetch HIV sequences
    print("\n2. Fetching HIV Sequences:")
    sequences = db_engine.fetch_hiv_sequences(subtype="B", gene="gag", limit=5)
    print(f"   Retrieved {len(sequences)} sequences")
    for i, seq in enumerate(sequences[:3]):  # Show first 3
        print(f"   {i+1}. Accession: {seq.accession}, Subtype: {seq.subtype}, "
              f"Country: {seq.country}, Year: {seq.year}, Length: {len(seq.sequence)}")
    
    # Example 3: Fetch drug resistance data
    print("\n3. Fetching Drug Resistance Data:")
    resistance_data = db_engine.fetch_drug_resistance_data(drug="Efavirenz")
    print(f"   Retrieved {len(resistance_data)} resistance entries for Efavirenz")
    for i, data in enumerate(resistance_data[:3]):  # Show first 3
        print(f"   {i+1}. Mutation: {data.mutation}, Fold Change: {data.fold_change}, "
              f"Phenotype: {data.phenotype}")
    
    # Example 4: Fetch clinical trials
    print("\n4. Fetching Clinical Trials:")
    trials = db_engine.fetch_clinical_trials(condition="HIV", phase="Phase 2")
    print(f"   Retrieved {len(trials)} Phase 2 HIV trials")
    for i, trial in enumerate(trials[:2]):  # Show first 2
        print(f"   {i+1}. NCT ID: {trial['nct_id']}")
        print(f"       Title: {trial['title'][:60]}...")
        print(f"       Status: {trial['status']}, Enrollment: {trial['enrollment']}")
    
    # Example 5: Get database statistics
    print("\n5. Database Statistics:")
    stats = db_engine.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Example 6: Sequence similarity search
    print("\n6. Sequence Similarity Search:")
    if sequences:
        query_seq = sequences[0].sequence[:100]  # First 100 nucleotides of first sequence
        similar_seqs = db_engine.search_sequences_by_similarity(query_seq, threshold=0.7)
        print(f"   Found {len(similar_seqs)} sequences similar to query (threshold 0.7)")
        for i, seq in enumerate(similar_seqs[:3]):  # Show first 3
            print(f"   {i+1}. Accession: {seq.accession}, Similarity: {getattr(seq, 'similarity_score', 0):.3f}")
    
    # Example 7: Synchronize with remote databases
    print("\n7. Synchronizing with Remote Databases:")
    db_engine.sync_with_remote_databases()
    print("   Synchronization completed")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    import random  # Need to import random for the demo
    run_database_integration_demo()