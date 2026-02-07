"""
Comprehensive tests for the integration engine module.
"""
import unittest
from src.genome_integration.integration_engine import IntegrationEngine, HostCell, IntegrationEvent, IntegrationOutcome


class TestIntegrationEngine(unittest.TestCase):
    """Test cases for the IntegrationEngine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engine = IntegrationEngine()

    def test_initialization(self):
        """Test that IntegrationEngine initializes correctly."""
        self.assertEqual(self.engine.integration_events, [])
        self.assertEqual(self.engine.cells, [])
        self.assertIn('promoter_regions', self.engine.integration_preferences)
        self.assertIn('transcription_units', self.engine.integration_preferences)
        self.assertIn('chr1', self.engine.chromosome_bias)

    def test_create_host_cell(self):
        """Test creating a host cell."""
        cell = self.engine.create_host_cell("CD4_T_CELL", 0.5)
        self.assertEqual(cell.cell_type, "CD4_T_CELL")
        self.assertEqual(cell.activation_state, 0.5)
        self.assertIsInstance(cell.chromatin_state, dict)
        self.assertIn(cell, self.engine.cells)

    def test_calculate_integration_success(self):
        """Test calculating integration success probability."""
        cell = self.engine.create_host_cell("CD4_T_CELL", 0.7)
        success_prob = self.engine._calculate_integration_success(cell)
        self.assertIsInstance(success_prob, float)
        self.assertGreaterEqual(success_prob, 0.0)
        self.assertLessEqual(success_prob, 0.95)  # Upper bound applied

    def test_calculate_latency_probability(self):
        """Test calculating latency probability."""
        cell = self.engine.create_host_cell("CD4_T_CELL", 0.3)
        site = ("chr1", 1000000)
        latency_prob = self.engine._calculate_latency_probability(cell, site)
        self.assertIsInstance(latency_prob, float)
        self.assertGreaterEqual(latency_prob, 0.0)
        self.assertLessEqual(latency_prob, 0.95)  # Upper bound applied

    def test_simulate_integration(self):
        """Test simulating an integration event."""
        cell = self.engine.create_host_cell("CD4_T_CELL", 0.5)
        viral_genome = "ATGCGATCGTAGCTAGCTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        
        event = self.engine.simulate_integration(cell, viral_genome, integration_time=0.0)
        self.assertIsInstance(event, IntegrationEvent)
        self.assertIn(event, self.engine.integration_events)
        self.assertEqual(event.host_cell, cell)
        self.assertEqual(event.viral_genome, viral_genome)
        self.assertIsInstance(event.outcome, IntegrationOutcome)
        self.assertIsInstance(event.integration_efficiency, float)
        self.assertIsInstance(event.latency_probability, float)
        self.assertEqual(event.timestamp, 0.0)

    def test_get_integration_summary(self):
        """Test getting integration summary."""
        summary = self.engine.get_integration_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("error", summary)
        self.assertEqual(summary["error"], "No integration events recorded")

        # Add a cell and simulate integration
        cell = self.engine.create_host_cell("CD4_T_CELL", 0.5)
        viral_genome = "ATGCGATCGTAGCTAGCTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        self.engine.simulate_integration(cell, viral_genome, integration_time=0.0)

        summary = self.engine.get_integration_summary()
        self.assertNotIn("error", summary)
        self.assertIn("total_events", summary)
        self.assertIn("successful_integrations", summary)
        self.assertIn("latent_integrations", summary)
        self.assertIn("failed_integrations", summary)
        self.assertIn("average_efficiency", summary)
        self.assertIn("cell_type_distribution", summary)
        self.assertIn("latent_reservoir_size", summary)


class TestHostCell(unittest.TestCase):
    """Test cases for the HostCell class."""

    def test_initialization(self):
        """Test that HostCell initializes correctly."""
        cell = HostCell(
            cell_id="test_cell",
            cell_type="CD4_T_CELL",
            activation_state=0.5,
            chromatin_state={"promoter_regions": 0.5, "gene_dense_regions": 0.3}
        )
        self.assertEqual(cell.cell_id, "test_cell")
        self.assertEqual(cell.cell_type, "CD4_T_CELL")
        self.assertEqual(cell.activation_state, 0.5)
        self.assertEqual(cell.chromatin_state, {"promoter_regions": 0.5, "gene_dense_regions": 0.3})
        self.assertEqual(cell.proviral_load, 0)
        self.assertFalse(cell.integrated_virus)
        self.assertFalse(cell.latent_state)
        self.assertIsNone(cell.integration_site)


class TestIntegrationEvent(unittest.TestCase):
    """Test cases for the IntegrationEvent class."""

    def test_initialization(self):
        """Test that IntegrationEvent initializes correctly."""
        cell = HostCell(
            cell_id="test_cell",
            cell_type="CD4_T_CELL",
            activation_state=0.5,
            chromatin_state={"promoter_regions": 0.5, "gene_dense_regions": 0.3}
        )
        event = IntegrationEvent(
            event_id="test_event",
            host_cell=cell,
            viral_genome="ATGC",
            integration_site=("chr1", 1000000),
            outcome=IntegrationOutcome.SUCCESSFUL_INTEGRATION,
            integration_efficiency=0.8,
            latency_probability=0.2,
            timestamp=1.0
        )
        self.assertEqual(event.event_id, "test_event")
        self.assertEqual(event.host_cell, cell)
        self.assertEqual(event.viral_genome, "ATGC")
        self.assertEqual(event.integration_site, ("chr1", 1000000))
        self.assertEqual(event.outcome, IntegrationOutcome.SUCCESSFUL_INTEGRATION)
        self.assertEqual(event.integration_efficiency, 0.8)
        self.assertEqual(event.latency_probability, 0.2)
        self.assertEqual(event.timestamp, 1.0)


if __name__ == '__main__':
    unittest.main()