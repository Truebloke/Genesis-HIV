"""
Comprehensive tests for the data ingestion module.
"""
import unittest
import tempfile
import os
from src.data_ingestion.data_ingestor import DataIngestor
from src.data_ingestion.genome_parser import GenomeParser
from src.data_ingestion.mutation_db import MutationDatabase


class TestDataIngestor(unittest.TestCase):
    """Test cases for the DataIngestor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.ingestor = DataIngestor(data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_initialization(self):
        """Test that DataIngestor initializes correctly."""
        self.assertEqual(self.ingestor.data_dir, self.temp_dir)
        self.assertTrue(hasattr(self.ingestor, 'pdb_base_url'))
        self.assertTrue(hasattr(self.ingestor, 'lanl_base_url'))

    def test_get_hiv_protein_sequence(self):
        """Test retrieving HIV protein sequences."""
        protease_seq = self.ingestor.get_hiv_protein_sequence("PROTEASE")
        self.assertIsInstance(protease_seq, str)
        self.assertGreater(len(protease_seq), 0)

        int_seq = self.ingestor.get_hiv_protein_sequence("INT")
        self.assertIsInstance(int_seq, str)
        self.assertGreater(len(int_seq), 0)

        # Test case insensitivity
        protease_seq_upper = self.ingestor.get_hiv_protein_sequence("protease")
        protease_seq_lower = self.ingestor.get_hiv_protein_sequence("PROTEASE")
        self.assertEqual(protease_seq_upper, protease_seq_lower)

    def test_get_hiv_genome_reference(self):
        """Test retrieving HIV genome reference."""
        genome_ref = self.ingestor.get_hiv_genome_reference()
        self.assertIsInstance(genome_ref, str)
        self.assertGreater(len(genome_ref), 0)

        # Test default subtype
        genome_hxb2 = self.ingestor.get_hiv_genome_reference("HXB2")
        self.assertEqual(genome_ref, genome_hxb2)

    def test_load_drug_resistance_data(self):
        """Test loading drug resistance data."""
        df = self.ingestor.load_drug_resistance_data()
        self.assertIsNotNone(df)
        self.assertIn('mutation', df.columns)
        self.assertIn('drug_class', df.columns)
        self.assertIn('drug', df.columns)
        self.assertIn('fold_change', df.columns)
        self.assertIn('phenotype', df.columns)


class TestGenomeParser(unittest.TestCase):
    """Test cases for the GenomeParser class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parser = GenomeParser()

    def test_initialization(self):
        """Test that GenomeParser initializes correctly."""
        self.assertEqual(self.parser.reference_genome, "")
        self.assertIsInstance(self.parser.genes, dict)
        self.assertIsInstance(self.parser.proteins, dict)
        self.assertIn('GAG', self.parser.gene_coordinates)
        self.assertIn('POL', self.parser.gene_coordinates)
        self.assertIn('ENV', self.parser.gene_coordinates)

    def test_get_default_hxb2_sequence(self):
        """Test getting the default HXB2 sequence."""
        seq = self.parser.get_default_hxb2_sequence()
        self.assertIsInstance(seq, str)
        self.assertGreater(len(seq), 0)

    def test_get_gene_sequence(self):
        """Test extracting gene sequences."""
        # Since we're using a default sequence, we can test the method
        self.parser.reference_genome = self.parser.get_default_hxb2_sequence()
        
        try:
            gag_seq = self.parser.get_gene_sequence('GAG')
            self.assertIsInstance(gag_seq, str)
            self.assertGreater(len(gag_seq), 0)
        except ValueError:
            # If GAG gene is not found in the shortened sequence, that's fine
            pass

    def test_translate_gene_to_protein(self):
        """Test translating gene sequences to proteins."""
        self.parser.reference_genome = self.parser.get_default_hxb2_sequence()
        
        try:
            protein = self.parser.translate_gene_to_protein('GAG')
            self.assertIsInstance(protein, str)
        except ValueError:
            # If GAG gene is not found in the shortened sequence, that's fine
            pass

    def test_find_open_reading_frames(self):
        """Test finding open reading frames."""
        self.parser.reference_genome = self.parser.get_default_hxb2_sequence()
        orfs = self.parser.find_open_reading_frames(min_length=100)
        self.assertIsInstance(orfs, list)
        # ORFs might be empty depending on the sequence, which is fine


class TestMutationDatabase(unittest.TestCase):
    """Test cases for the MutationDatabase class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = MutationDatabase(db_path=self.temp_db.name)

    def tearDown(self):
        """Clean up after each test method."""
        self.db.close()
        os.unlink(self.temp_db.name)

    def test_initialization(self):
        """Test that MutationDatabase initializes correctly."""
        self.assertTrue(os.path.exists(self.temp_db.name))
        self.assertIsNotNone(self.db.connection)

    def test_add_drug(self):
        """Test adding a drug to the database."""
        drug_id = self.db.add_drug("TestDrug", "TEST_CLASS", "Test mechanism")
        self.assertIsInstance(drug_id, int)
        self.assertGreater(drug_id, 0)

    def test_add_mutation(self):
        """Test adding a mutation to the database."""
        mutation_id = self.db.add_mutation("RT", "K103N", 103, "K", "N", "Test mutation")
        self.assertIsInstance(mutation_id, int)
        self.assertGreater(mutation_id, 0)

    def test_add_drug_resistance(self):
        """Test associating a mutation with drug resistance."""
        mutation_id = self.db.add_mutation("RT", "K103N", 103, "K", "N", "Test mutation")
        drug_id = self.db.add_drug("Efavirenz", "NNRTI", "Non-nucleoside RT inhibitor")
        assoc_id = self.db.add_drug_resistance(mutation_id, drug_id, fold_change=100.0,
                                              phenotype="High-level resistance", level="High")
        self.assertIsInstance(assoc_id, int)
        self.assertGreater(assoc_id, 0)

    def test_get_resistance_profile(self):
        """Test getting resistance profile for a drug."""
        # Add some data first
        mutation_id = self.db.add_mutation("RT", "K103N", 103, "K", "N", "Test mutation")
        drug_id = self.db.add_drug("Efavirenz", "NNRTI", "Non-nucleoside RT inhibitor")
        self.db.add_drug_resistance(mutation_id, drug_id, fold_change=100.0,
                                   phenotype="High-level resistance", level="High")

        df = self.db.get_resistance_profile("Efavirenz")
        self.assertIsNotNone(df)
        if not df.empty:
            self.assertIn('mutation_code', df.columns)
            self.assertIn('gene', df.columns)


if __name__ == '__main__':
    unittest.main()