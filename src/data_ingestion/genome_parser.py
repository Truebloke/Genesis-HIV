"""
Genome parser module for HIV HXB2 reference sequence.

This module handles parsing and manipulation of the HIV genome reference sequence,
including annotation of genes, ORFs, and other genomic features.
"""

import re
from typing import Dict, List, Tuple
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


class GenomeParser:
    """
    A class to parse and manipulate HIV genome sequences.
    
    This includes:
    - Reading and validating HIV genome sequences
    - Identifying genes and ORFs
    - Translating nucleotide sequences to proteins
    - Detecting mutations and variations
    """
    
    def __init__(self, reference_genome_path=None):
        """
        Initialize the GenomeParser.
        
        Args:
            reference_genome_path (str, optional): Path to reference genome file
        """
        self.reference_genome = ""
        self.genes = {}
        self.proteins = {}
        
        # Define HIV gene coordinates in HXB2 reference
        self.gene_coordinates = {
            'LTR': [(1, 634), (9053, 9687)],
            'GAG': [790, 2292],  # p17, p24, p2, p7, p1
            'POL': [2085, 5096],  # PR, RT, RNase H, IN
            'ENV': [6225, 8795],  # gp120, gp41
            'TAT': [5831, 5972],
            'REV': [5970, 6045, 8379, 8469],
            'VPU': [6062, 6310],
            'VIF': [5041, 5619],
            'VPR': [5559, 5850],
            'NEF': [8797, 9052]
        }
        
        if reference_genome_path:
            self.load_reference_genome(reference_genome_path)
    
    def load_reference_genome(self, path: str) -> None:
        """
        Load the reference HIV genome from a file.
        
        Args:
            path (str): Path to the genome file (FASTA format)
        """
        try:
            record = SeqIO.read(path, "fasta")
            self.reference_genome = str(record.seq)
            print(f"Loaded reference genome of length {len(self.reference_genome)}")
        except FileNotFoundError:
            print(f"Reference genome file not found: {path}")
            print("Using default HXB2 sequence...")
            self.reference_genome = self.get_default_hxb2_sequence()
    
    def get_default_hxb2_sequence(self) -> str:
        """
        Return the default HXB2 reference sequence.
        
        Returns:
            str: HXB2 reference genome sequence
        """
        # This is a shortened version of the HXB2 reference sequence
        # In practice, we would load the full sequence from a file
        return ("AACAAAGACTGTATACATACAAGCAGCTATTCAGCTTGAGGCTCAGCAGCTCACCAGAATGACTTTAACAATGGAAAAGGAAGGACACCCACTGCTTAGAGAATGTAACAGAAAAATTATAAAACATTGTGAAACCACCCTGGCCGTCAGTACAATTAAAAAGCAGAGGGAAGATACAGAGATAGGACCCGGCACAGGGAAGGTCCCATCCTATGAATGGATGATCTGGGACCTTTGGGAGTGGGATCGGGTAGAGGTAGAACATGAGAATGGAGCTGTAGCCCATAGCATCAGTCTAGAATGGAAGAAAAAATTGATAGCAGAAATAGTACCTCTAACAGAAGAAAGAAGAGCCAGAGATGCTAGCAATGCAAGAGTTTTGGGAGCTATCTGGACAATAGGAATAGCCTATGCTGGTGTAGAAATCAAGGGAACCATTGAGGTGCTGGGAAAATGGCTAGGAAATATGACCATAGTACCAAGAATAAAACAAAGAGTAGAAGTACCACTGCCCATAGGGACATGGAGCCAGTAGGCAGAAAGCATCTGGAACTGGGGGATATGGGGGGCAGGAGACAGCAGCATGCTGCGGGATCTGATCACCTCCCTGGGTCGCGACCCCTCTCAAAAGTAGTCATGAAAGTCTGGATACCAGGAAAGTCAACTTTAGCCTTGTGGCAGCTAACAAGAGGGGAAAGGAGGAGAAGTAGACTACCTCCCTCCAGGAGAAGATATAGCAGGGAAGATGCTTTTCTTGTACCTTAGAGGGAAAGGATCACCATCCTGGTTCAGATGACCTACAGAGAATATGTGAAATGGCCACATGCCCATCCTGATGGGCCAAAATGATAGGCCCTGTAAAGCAGAGAATTAGAGGACCTGTAGATCTTCCAGCCTCAGGATGGGGGGGAGCTGATTTCTCCCAGAGCCTTACTCTAAGACCTCTACTCAAGAAGCTATATTCCAAAGTTCCTGGGAATCTCGGGACTGCTTGTTTTCAAGGTGCTTTCTTAGTGAAGGGGGGGATCTGAGTGTGGGATGGGGTTCCTTTGTGTGGTGGTGGAAATGTAGATTATGACATGTATAGTTATATAGGCTCTCTCTATGTGTGTGTGGACATGTAA")
    
    def get_gene_sequence(self, gene_name: str) -> str:
        """
        Extract the nucleotide sequence for a specific gene.
        
        Args:
            gene_name (str): Name of the gene (e.g., 'GAG', 'POL', 'ENV')
            
        Returns:
            str: Nucleotide sequence of the gene
        """
        if gene_name.upper() not in self.gene_coordinates:
            raise ValueError(f"Gene {gene_name} not found in reference")
        
        coords = self.gene_coordinates[gene_name.upper()]
        gene_seq = ""
        
        if isinstance(coords[0], tuple):  # Handle multi-part genes like REV
            for start, end in coords:
                gene_seq += self.reference_genome[start-1:end]  # Convert to 0-indexed
        else:
            start, end = coords[0], coords[1]
            gene_seq = self.reference_genome[start-1:end]  # Convert to 0-indexed
        
        return gene_seq
    
    def translate_gene_to_protein(self, gene_name: str) -> str:
        """
        Translate a gene's nucleotide sequence to its protein sequence.
        
        Args:
            gene_name (str): Name of the gene
            
        Returns:
            str: Protein sequence of the gene
        """
        gene_seq = self.get_gene_sequence(gene_name)
        nucleotide_seq = Seq(gene_seq)
        protein_seq = nucleotide_seq.translate(to_stop=True)
        return str(protein_seq)
    
    def find_open_reading_frames(self, min_length: int = 100) -> List[Tuple[int, int, str]]:
        """
        Find potential open reading frames in the genome.
        
        Args:
            min_length (int): Minimum length of ORF to consider
            
        Returns:
            List[Tuple[int, int, str]]: List of ORFs as (start, end, sequence) tuples
        """
        orfs = []
        codon_start = 'ATG'
        codon_stop = ['TAA', 'TAG', 'TGA']
        
        # Search in all three forward reading frames
        for frame in range(3):
            pos = frame
            while pos < len(self.reference_genome) - 2:
                if self.reference_genome[pos:pos+3] == codon_start:
                    orf_start = pos
                    pos += 3
                    while pos < len(self.reference_genome) - 2:
                        codon = self.reference_genome[pos:pos+3]
                        if codon in codon_stop:
                            orf_end = pos + 3
                            if (orf_end - orf_start) >= min_length:
                                orfs.append((orf_start, orf_end, self.reference_genome[orf_start:orf_end]))
                            break
                        pos += 3
                else:
                    pos += 3
        
        return orfs
    
    def detect_mutations(self, query_sequence: str) -> Dict[str, List[Tuple[int, str, str]]]:
        """
        Compare a query sequence to the reference and detect mutations.
        
        Args:
            query_sequence (str): Query sequence to compare
            
        Returns:
            Dict[str, List[Tuple[int, str, str]]]: Dictionary of mutations
        """
        mutations = {'substitutions': [], 'insertions': [], 'deletions': []}
        
        ref_len = len(self.reference_genome)
        qry_len = len(query_sequence)
        min_len = min(ref_len, qry_len)
        
        # Find substitutions
        for i in range(min_len):
            if self.reference_genome[i] != query_sequence[i]:
                mutations['substitutions'].append((i+1, self.reference_genome[i], query_sequence[i]))  # 1-indexed
        
        # Find insertions/deletions
        if qry_len > ref_len:
            # Insertions in query
            for i in range(ref_len, qry_len):
                mutations['insertions'].append((i+1, query_sequence[i]))
        elif ref_len > qry_len:
            # Deletions in query
            for i in range(qry_len, ref_len):
                mutations['deletions'].append((i+1, self.reference_genome[i]))
        
        return mutations
    
    def annotate_sequence(self, sequence: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Annotate a sequence with gene positions and features.
        
        Args:
            sequence (str): Sequence to annotate
            
        Returns:
            Dict[str, List[Tuple[int, int, str]]]: Annotation dictionary
        """
        annotations = {}
        
        for gene_name, coords in self.gene_coordinates.items():
            if isinstance(coords[0], tuple):  # Multi-part gene
                gene_positions = []
                for start, end in coords:
                    if start <= len(sequence) and end <= len(sequence):
                        gene_positions.append((start, end, gene_name))
                if gene_positions:
                    annotations[gene_name] = gene_positions
            else:  # Single-part gene
                start, end = coords[0], coords[1]
                if start <= len(sequence) and end <= len(sequence):
                    annotations[gene_name] = [(start, end, gene_name)]
        
        return annotations


if __name__ == "__main__":
    # Example usage
    parser = GenomeParser()
    
    print("HIV Genome Parser initialized")
    print(f"Reference genome length: {len(parser.reference_genome)}")
    
    # Extract and translate a gene
    gag_seq = parser.get_gene_sequence('GAG')
    print(f"GAG gene length: {len(gag_seq)} nucleotides")
    
    gag_protein = parser.translate_gene_to_protein('GAG')
    print(f"GAG protein length: {len(gag_protein)} amino acids")
    print(f"GAG protein sequence: {gag_protein[:50]}...")  # Show first 50 chars
    
    # Find ORFs
    orfs = parser.find_open_reading_frames(min_length=150)
    print(f"Found {len(orfs)} potential ORFs with length >= 150")
    
    # Annotate the reference genome
    annotations = parser.annotate_sequence(parser.reference_genome)
    print(f"Annotated genes: {list(annotations.keys())}")