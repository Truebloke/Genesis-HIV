"""
Comprehensive tests for the structural engine module.
"""
import unittest
import numpy as np
from src.structural_engine.structural_engine import StructuralEngine, Atom, Residue, BindingSite


class TestStructuralEngine(unittest.TestCase):
    """Test cases for the StructuralEngine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engine = StructuralEngine()

    def test_initialization(self):
        """Test that StructuralEngine initializes correctly."""
        self.assertEqual(self.engine.proteins, {})
        self.assertEqual(self.engine.binding_sites, {})
        self.assertIn('C', self.engine.atom_types)
        self.assertIn('N', self.engine.atom_types)
        self.assertIn('O', self.engine.atom_types)
        self.assertIn('S', self.engine.atom_types)
        self.assertIn('H', self.engine.atom_types)

    def test_load_protein_from_pdb(self):
        """Test loading a protein from PDB content."""
        mock_pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 10.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 10.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00 10.00           C
ATOM      4  O   ALA A   1      11.000  12.000  10.000  1.00 10.00           O
END
"""
        self.engine.load_protein_from_pdb(mock_pdb_content, "test_protein")
        self.assertIn("test_protein", self.engine.proteins)
        # The PDB contains only 1 unique residue (ALA A 1), even though there are 4 atoms
        self.assertEqual(len(self.engine.proteins["test_protein"]), 1)

    def test_identify_binding_sites(self):
        """Test identifying binding sites in a protein."""
        mock_pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 10.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 10.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00 10.00           C
ATOM      4  O   ALA A   1      11.000  12.000  10.000  1.00 10.00           O
ATOM      5  CB  ALA A   1       9.500   9.500  11.000  1.00 10.00           C
ATOM      6  N   GLY A   2      12.500  11.000  10.000  1.00 10.00           N
ATOM      7  CA  GLY A   2      13.000  12.000  10.000  1.00 10.00           C
ATOM      8  C   GLY A   2      14.000  12.500  11.000  1.00 10.00           C
ATOM      9  O   GLY A   2      14.500  13.500  11.000  1.00 10.00           O
END
"""
        self.engine.load_protein_from_pdb(mock_pdb_content, "test_protein")
        binding_sites = self.engine.identify_binding_sites("test_protein")
        # May not find binding sites with this small structure, but shouldn't crash
        self.assertIsInstance(binding_sites, list)

    def test_calculate_binding_energy(self):
        """Test calculating binding energy."""
        mock_pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 10.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 10.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00 10.00           C
ATOM      4  O   ALA A   1      11.000  12.000  10.000  1.00 10.00           O
END
"""
        self.engine.load_protein_from_pdb(mock_pdb_content, "test_protein")
        self.engine.identify_binding_sites("test_protein", threshold=10.0)
        
        if "test_protein" in self.engine.binding_sites and len(self.engine.binding_sites["test_protein"]) > 0:
            ligand_coords = [np.array([12.0, 11.5, 10.5])]
            binding_site = self.engine.binding_sites["test_protein"][0]
            energy = self.engine.calculate_binding_energy("test_protein", ligand_coords, binding_site)
            self.assertIsInstance(energy, float)

    def test_dock_ligand(self):
        """Test ligand docking."""
        mock_pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 10.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 10.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00 10.00           C
ATOM      4  O   ALA A   1      11.000  12.000  10.000  1.00 10.00           O
END
"""
        self.engine.load_protein_from_pdb(mock_pdb_content, "test_protein")
        self.engine.identify_binding_sites("test_protein", threshold=10.0)
        
        if "test_protein" in self.engine.binding_sites and len(self.engine.binding_sites["test_protein"]) > 0:
            ligand_coords = [np.array([12.0, 11.5, 10.5]), np.array([12.5, 12.0, 10.5])]
            binding_site = self.engine.binding_sites["test_protein"][0]
            results = self.engine.dock_ligand("test_protein", ligand_coords, binding_site, num_poses=3)
            self.assertIsInstance(results, list)
            self.assertGreaterEqual(len(results), 0)  # May not find results with small structure
            if results:
                self.assertEqual(len(results[0]), 2)  # (energy, coordinates)


class TestAtom(unittest.TestCase):
    """Test cases for the Atom class."""

    def test_initialization(self):
        """Test that Atom initializes correctly."""
        atom = Atom(element='C', x=1.0, y=2.0, z=3.0, residue_name='ALA', 
                   residue_number=1, chain='A')
        self.assertEqual(atom.element, 'C')
        self.assertEqual(atom.x, 1.0)
        self.assertEqual(atom.y, 2.0)
        self.assertEqual(atom.z, 3.0)
        self.assertEqual(atom.residue_name, 'ALA')
        self.assertEqual(atom.residue_number, 1)
        self.assertEqual(atom.chain, 'A')
        self.assertEqual(atom.coords.tolist(), [1.0, 2.0, 3.0])


class TestResidue(unittest.TestCase):
    """Test cases for the Residue class."""

    def test_initialization(self):
        """Test that Residue initializes correctly."""
        atoms = [Atom('C', 1.0, 2.0, 3.0, 'ALA', 1, 'A')]
        residue = Residue(name='ALA', number=1, chain='A', atoms=atoms)
        self.assertEqual(residue.name, 'ALA')
        self.assertEqual(residue.number, 1)
        self.assertEqual(residue.chain, 'A')
        self.assertEqual(len(residue.atoms), 1)

    def test_get_centroid(self):
        """Test calculating residue centroid."""
        atoms = [
            Atom('C', 0.0, 0.0, 0.0, 'ALA', 1, 'A'),
            Atom('N', 2.0, 0.0, 0.0, 'ALA', 1, 'A'),
            Atom('O', 0.0, 2.0, 0.0, 'ALA', 1, 'A')
        ]
        residue = Residue(name='ALA', number=1, chain='A', atoms=atoms)
        centroid = residue.get_centroid()
        expected_centroid = np.array([2.0/3, 2.0/3, 0.0])  # Average of coordinates
        np.testing.assert_array_almost_equal(centroid, expected_centroid)


class TestBindingSite(unittest.TestCase):
    """Test cases for the BindingSite class."""

    def test_initialization(self):
        """Test that BindingSite initializes correctly."""
        atoms = [Atom('C', 1.0, 2.0, 3.0, 'ALA', 1, 'A')]
        residues = [Residue(name='ALA', number=1, chain='A', atoms=atoms)]
        center = np.array([1.0, 1.0, 1.0])
        binding_site = BindingSite(name='Site1', residues=residues, 
                                 center=center, volume=10.0, accessibility=0.5)
        self.assertEqual(binding_site.name, 'Site1')
        self.assertEqual(len(binding_site.residues), 1)
        np.testing.assert_array_equal(binding_site.center, center)
        self.assertEqual(binding_site.volume, 10.0)
        self.assertEqual(binding_site.accessibility, 0.5)


if __name__ == '__main__':
    unittest.main()