#!/usr/bin/env python3
"""
Generate training data for T5 model fine-tuning
Converts natural language queries to API endpoints
"""
from pathlib import Path
import json
import logging
import pandas as pd
from urllib.parse import quote

from config.logging_config import configure_logger


class TrainingDataGenerator:
    def __init__(self, bridge_data_path=None, log_level="INFO"):
        self.logger = logging.getLogger(__name__)
        configure_logger(log_level=log_level, logger=self.logger)

        self.training_examples = []
        self.counties = [
            "Columbia",
            "Dutchess",
            "Orange",
            "Putnam",
            "Rockland",
            "Ulster",
            "Westchester",
        ]

        self._load_features()
        self._load_bridge_bins(bridge_data_path)

    def _load_features(self):
        """Load CARRIED and CROSSED features from extracted_features.py"""
        try:
            from extracted_features import CARRIED, CROSSED

            self.carried = CARRIED
            self.crossed = CROSSED
            self.logger.info(f"Loaded {len(CARRIED)} carried and {len(CROSSED)} crossed features")
        except ImportError as e:
            self.logger.error(f"Could not import extracted_features.py: {e}")
            self.logger.error("Run extract_features.py first from project root")
            raise

    def _load_bridge_bins(self, data_path=None):
        """Load real bridge IDs from CSV data"""
        if data_path is None:
            script_dir = Path(__file__).parent
            data_path = script_dir.parent / "data" / "bridge_data.csv"

        df = pd.read_csv(data_path)
        self.bridge_bins = df["bin"].dropna().astype(str).unique()[:10].tolist()
        self.logger.info(f"Loaded {len(self.bridge_bins)} bridge IDs from data")

    def add_examples(self, templates, output):
        """Helper to add multiple template variations with same output."""
        for template in templates:
            self.training_examples.append({"input": f"translate to api: {template}", "output": output})

    def generate_basic_queries(self):
        """Generate basic count and list queries"""
        self.add_examples(
            [
                "How many bridges are there?",
                "Total number of bridges",
                "Count all bridges",
                "What's the total bridge count?",
                "How many bridges in the database?",
                "Give me the bridge count",
                "Total bridges",
                "Number of bridges",
                "How many bridges do we have?",
                "Bridge count",
            ],
            "/api/bridges/count",
        )

        self.add_examples(
            [
                "Show me bridges",
                "List all bridges",
                "Get bridges",
                "Display bridges",
                "Show bridge list",
            ],
            "/api/bridges",
        )

    def generate_county_queries(self):
        """Generate county-based queries"""
        for county in self.counties:
            for template in [
                "Show me bridges in {county}",
                "Bridges in {county} county",
                "Find bridges in {county}",
                "List bridges in {county}",
                "What bridges are in {county}?",
                "Get {county} bridges",
                "{county} county bridges",
                "Show {county} bridges",
            ]:
                self.training_examples.append(
                    {
                        "input": f"translate to api: {template.format(county=county)}",
                        "output": f"/api/bridges/search?county={county}",
                    }
                )

            for template in [
                "Bridge with most spans in {county}",
                "Highest span count in {county}",
                "Top bridge by spans in {county}",
                "{county} bridge with maximum spans",
                "Which {county} bridge has most spans?",
            ]:
                self.training_examples.append(
                    {
                        "input": f"translate to api: {template.format(county=county)}",
                        "output": f"/api/bridges/search?county={county}&sort=spans&order=desc&limit=1",
                    }
                )

            for n in [3, 5, 10]:
                self.training_examples.append(
                    {
                        "input": f"translate to api: Top {n} bridges by spans in {county}",
                        "output": f"/api/bridges/search?county={county}&sort=spans&order=desc&limit={n}",
                    }
                )
                self.training_examples.append(
                    {
                        "input": f"translate to api: Show me {n} bridges with most spans in {county}",
                        "output": f"/api/bridges/search?county={county}&sort=spans&order=desc&limit={n}",
                    }
                )

    def generate_span_queries(self):
        """Generate span-based queries with diverse number examples"""
        # Include small, medium, and large numbers for better generalization
        span_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50, 100]
        for spans in span_values:
            self.add_examples(
                [
                    f"Bridges with {spans} spans",
                    f"Show bridges with exactly {spans} spans",
                    f"Find {spans} span bridges",
                    f"{spans}-span bridges",
                    f"Bridges that have {spans} spans",
                ],
                f"/api/bridges/search?spans={spans}",
            )

        # Include diverse numbers for "more than" queries
        min_span_values = [2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        for min_span in min_span_values:
            self.add_examples(
                [
                    f"Bridges with more than {min_span} spans",
                    f"Bridges with at least {min_span+1} spans",
                    f"Show bridges with {min_span}+ spans",
                    f"Find bridges with over {min_span} spans",
                ],
                f"/api/bridges/search?min_spans={min_span+1}",
            )

        for max_span in [2, 3, 4]:
            self.add_examples(
                [
                    f"Bridges with less than {max_span+1} spans",
                    f"Bridges with at most {max_span} spans",
                    f"Bridges with {max_span} or fewer spans",
                ],
                f"/api/bridges/search?max_spans={max_span}",
            )

    def generate_feature_queries(self):
        """Generate carried/crossed feature queries"""
        for feature in self.carried:
            # URL encode features with spaces
            encoded_feature = quote(feature, safe='')
            self.add_examples(
                [
                    f"Bridges that carry {feature}",
                    f"Bridges carrying {feature}",
                    f"Find bridges carrying {feature}",
                    f"Show {feature} bridges",
                    f"Bridges with {feature}",
                    f"Search bridges carrying {feature}",
                ],
                f"/api/bridges/search?carried={encoded_feature}",
            )

        for feature in self.crossed:
            # URL encode features with spaces
            encoded_feature = quote(feature, safe='')
            self.add_examples(
                [
                    f"Bridges that cross {feature}",
                    f"Bridges crossing {feature}",
                    f"Find bridges crossing {feature}",
                    f"Bridges over {feature}",
                    f"Show bridges that cross {feature}",
                    f"Search bridges crossing {feature}",
                ],
                f"/api/bridges/search?crossed={encoded_feature}",
            )

    def generate_multifilter_queries(self):
        """Generate queries with multiple filters"""
        # Include diverse numbers for county+span combinations
        min_span_values = [2, 3, 4, 5, 10, 15, 20, 50]
        for county in self.counties:
            for min_span in min_span_values:
                self.add_examples(
                    [
                        f"Bridges in {county} with more than {min_span} spans",
                        f"{county} bridges with at least {min_span+1} spans",
                        f"Find bridges in {county} with {min_span}+ spans",
                        f"Show {county} bridges with over {min_span} spans",
                    ],
                    f"/api/bridges/search?county={county}&min_spans={min_span+1}",
                )

        for county in self.counties[:3]:
            for feature in self.carried[:3]:
                encoded_feature = quote(feature, safe='')
                self.add_examples(
                    [
                        f"{feature} bridges in {county}",
                        f"Bridges carrying {feature} in {county}",
                        f"Find {county} bridges carrying {feature}",
                    ],
                    f"/api/bridges/search?county={county}&carried={encoded_feature}",
                )

        for county in self.counties[:3]:
            for feature in self.crossed[:3]:
                encoded_feature = quote(feature, safe='')
                self.add_examples(
                    [
                        f"Bridges crossing {feature} in {county}",
                        f"{county} bridges over {feature}",
                        f"Find {county} bridges crossing {feature}",
                    ],
                    f"/api/bridges/search?county={county}&crossed={encoded_feature}",
                )

    def generate_statistics_queries(self):
        """Generate statistical aggregation queries"""
        self.add_examples(
            [
                "Bridge statistics by county",
                "Group bridges by county",
                "Show county statistics",
                "Bridge count per county",
                "County breakdown",
                "Statistics per county",
            ],
            "/api/bridges/group-by-county",
        )

        self.add_examples(
            [
                "Bridge statistics by span count",
                "Group bridges by spans",
                "Show span statistics",
                "Bridge count per span number",
                "Span breakdown",
            ],
            "/api/bridges/group-by-spans",
        )

    def generate_sorting_queries(self):
        """Generate sorting and limiting queries"""
        self.add_examples(
            [
                "Bridge with most spans",
                "Maximum span count",
                "Highest span bridge",
                "Bridge with maximum spans",
                "Which bridge has most spans?",
            ],
            "/api/bridges/search?sort=spans&order=desc&limit=1",
        )

        self.add_examples(
            [
                "Bridge with fewest spans",
                "Minimum span count",
                "Lowest span bridge",
                "Bridge with minimum spans",
                "Which bridge has least spans?",
            ],
            "/api/bridges/search?sort=spans&order=asc&limit=1",
        )

    def generate_inspection_queries(self):
        """Generate inspection-related queries"""
        self.add_examples(
            [
                "Show inspections",
                "List all inspections",
                "Get inspections",
                "Recent inspections",
                "All inspections",
            ],
            "/api/inspections",
        )

        for bin_id in self.bridge_bins:
            self.add_examples(
                [
                    f"Inspections for bridge {bin_id}",
                    f"Show inspections for {bin_id}",
                    f"Get bridge {bin_id} inspections",
                    f"Inspection history for {bin_id}",
                    f"Find inspections of bridge {bin_id}",
                ],
                f"/api/inspections/{bin_id}",
            )

    def generate_complex_queries(self):
        """Generate complex multi-parameter queries"""
        # Use actual features from data
        self.add_examples(
            [
                "Orange county bridges carrying 84i with more than 4 spans",
                "Find bridges in Orange carrying 84i with at least 5 spans",
            ],
            "/api/bridges/search?county=Orange&carried=84i&min_spans=5",
        )
        
        self.add_examples(
            [
                "Orange county bridges on interstate with more than 4 spans",
                "Find bridges in Orange on interstate with at least 5 spans",
            ],
            "/api/bridges/search?county=Orange&carried=i&min_spans=5",
        )

        self.add_examples(
            [
                "Westchester bridges crossing river with at least 3 spans",
                "Find Westchester bridges over river with 3+ spans",
            ],
            "/api/bridges/search?county=Westchester&crossed=river&min_spans=3",
        )

        self.add_examples(
            [
                "Find top 5 bridges by span count",
                "Show 5 bridges with most spans",
                "Top 5 highest span bridges",
            ],
            "/api/bridges/search?sort=spans&order=desc&limit=5",
        )

        self.add_examples(
            [
                "Show me the 10 bridges with most spans",
                "Top 10 bridges by spans",
                "Find 10 highest span count bridges",
            ],
            "/api/bridges/search?sort=spans&order=desc&limit=10",
        )

    def generate_all(self):
        """Generate all training examples"""
        self.logger.info("Generating training examples...")

        self.generate_basic_queries()
        self.generate_county_queries()
        self.generate_span_queries()
        self.generate_feature_queries()
        self.generate_multifilter_queries()
        self.generate_statistics_queries()
        self.generate_sorting_queries()
        self.generate_inspection_queries()
        self.generate_complex_queries()

        self.logger.info(f"Generated {len(self.training_examples)} total examples")

    def save(self, output_path="training_data.json"):
        """Save training examples to JSON file"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.training_examples, f, indent=2)

            self.logger.info(f"Saved {len(self.training_examples)} training examples to {output_path}")
        except IOError as e:
            self.logger.error(f"Error writing to {output_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
            raise


if __name__ == "__main__":
    # Save to src/data/ directory
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "data" / "training_data.json"
    
    generator = TrainingDataGenerator()
    generator.logger.info(f"Will save training data to: {output_path}")
    generator.generate_all()
    generator.save(str(output_path))
    generator.logger.info("Training data generation complete!")
