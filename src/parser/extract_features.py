#!/usr/bin/env python3
"""
Extract unique carried/crossed features from bridge_data.csv
to use in training data generation

Extracts SEARCHABLE KEYWORDS for SQL LIKE pattern matching
"""
from pathlib import Path
from config.logging_config import configure_logger
import re
import logging
from collections import Counter
import pandas as pd


class FeatureExtractor:
    def __init__(self, csv_path, log_level="INFO"):
        self.csv_path = csv_path
        self.features = None

        self.logger = logging.getLogger(__name__)
        configure_logger(log_level=log_level)

    def extract_searchable_keyword(self, text):
        """
        Extract the searchable keyword from bridge data strings.

        Examples:
            "987G 987G87011089" -> "987g"
            "100  100 87011011" -> "100"
            "684I 684I84021016" -> "684i"
            "WEST AVE." -> "west ave."
            "MOODNA CREEK" -> "moodna creek"
            "EB287toEB100+119" -> "eb287toeb100+119"
        """
        text = str(text).strip().lower()

        ref_pattern = r"^(9[0-8][0-9][a-z])\s+\1\d+$"
        match = re.match(ref_pattern, text)
        if match:
            return match.group(1)

        route_pattern = r"^(\d+[a-z]?)\s+\1\s+\d+$"
        match = re.match(route_pattern, text)
        if match:
            return match.group(1)

        standalone_ref = r"^(9[0-8][0-9][a-z])\s+\1$"
        match = re.match(standalone_ref, text)
        if match:
            return match.group(1)

        if re.match(r"^9[0-8][0-9][a-z]", text):
            return text[:4]

        simple_route = r"^(\d+[a-z]?)\s+"
        match = re.match(simple_route, text)
        if match:
            return match.group(1)

        return text

    def normalize_for_like(self, text):
        """Normalize text for SQL LIKE matching."""
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def extract_features(self):
        """Extract searchable keywords from carried and crossed columns."""
        try:
            self.logger.info(f"Reading {self.csv_path}...")
            df = pd.read_csv(self.csv_path)
            self.logger.info(f"Total bridges: {len(df)}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.csv_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error(f"File is empty: {self.csv_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
            raise

        try:
            carried_keywords = Counter()
            carried_examples = {}

            if "carried" not in df.columns:
                self.logger.warning("'carried' column not found in CSV")
            else:
                for item in df["carried"].dropna():
                    try:
                        original = str(item).strip()
                        keyword = self.extract_searchable_keyword(original)
                        keyword = self.normalize_for_like(keyword)

                        if keyword and len(keyword) > 0:
                            carried_keywords[keyword] += 1
                            if keyword not in carried_examples:
                                carried_examples[keyword] = original
                    except Exception as e:
                        self.logger.debug(f"Error processing carried item '{item}': {e}")
                        continue

            crossed_keywords = Counter()
            crossed_examples = {}

            if "crossed" not in df.columns:
                self.logger.warning("'crossed' column not found in CSV")
            else:
                for item in df["crossed"].dropna():
                    try:
                        original = str(item).strip()
                        keyword = self.extract_searchable_keyword(original)
                        keyword = self.normalize_for_like(keyword)

                        if keyword and len(keyword) > 0:
                            crossed_keywords[keyword] += 1
                            if keyword not in crossed_examples:
                                crossed_examples[keyword] = original
                    except Exception as e:
                        self.logger.debug(f"Error processing crossed item '{item}': {e}")
                        continue

            top_carried = [item[0] for item in carried_keywords.most_common(100)]
            top_crossed = [item[0] for item in crossed_keywords.most_common(100)]

            self.logger.info(f"Extracted {len(carried_keywords)} unique carried keywords")
            self.logger.info(f"Extracted {len(crossed_keywords)} unique crossed keywords")

            self.features = {
                "carried_keywords": top_carried,
                "crossed_keywords": top_crossed,
                "carried_counts": carried_keywords,
                "crossed_counts": crossed_keywords,
                "carried_examples": carried_examples,
                "crossed_examples": crossed_examples,
                "total_carried": len(carried_keywords),
                "total_crossed": len(crossed_keywords),
            }

            return self.features

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise

    def display_features(self):
        """Display extracted searchable keywords."""
        if not self.features:
            self.logger.error("No features extracted yet. Call extract_features() first.")
            return

        try:
            self.logger.info(f"\nCARRIED: Top {len(self.features['carried_keywords'])} keywords")
            for i, kw in enumerate(self.features["carried_keywords"][:10], 1):
                self.logger.info(f"  {i}. {kw} ({self.features['carried_counts'][kw]})")

            self.logger.info(f"\nCROSSED: Top {len(self.features['crossed_keywords'])} keywords")
            for i, kw in enumerate(self.features["crossed_keywords"][:10], 1):
                self.logger.info(f"  {i}. {kw} ({self.features['crossed_counts'][kw]})")

            self.logger.info(
                f"\nTotal: {self.features['total_carried']} carried, {self.features['total_crossed']} crossed"
            )
        except Exception as e:
            self.logger.error(f"Error displaying features: {e}")
            raise

    def save_features_file(self, output_path="extracted_features.py"):
        """Save searchable keywords as Python file for use in training data generator."""
        if not self.features:
            self.logger.error("No features extracted yet. Call extract_features() first.")
            return

        try:
            self.logger.info(f"Saving features to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write('"""\n')
                f.write("Extracted SEARCHABLE KEYWORDS from actual bridge data\n")
                f.write("These are normalized for SQL LIKE pattern matching\n")
                f.write('"""\n\n')

                f.write("# Top 100 most common searchable 'carried' keywords\n")
                f.write("# Use these in: WHERE carried LIKE '%{keyword}%'\n")
                f.write("CARRIED = [\n")
                for keyword in self.features["carried_keywords"]:
                    example = self.features["carried_examples"].get(keyword, "")
                    f.write(f'    "{keyword}",  # e.g., "{example}"\n')
                f.write("]\n\n")

                f.write("# Top 100 most common searchable 'crossed' keywords\n")
                f.write("# Use these in: WHERE crossed LIKE '%{keyword}%'\n")
                f.write("CROSSED = [\n")
                for keyword in self.features["crossed_keywords"]:
                    example = self.features["crossed_examples"].get(keyword, "")
                    f.write(f'    "{keyword}",  # e.g., "{example}"\n')
                f.write("]\n\n")

                f.write(
                    f"# Total unique keywords: {self.features['total_carried']} carried, {self.features['total_crossed']} crossed\n"
                )

            self.logger.info(f"Successfully saved searchable keywords to {output_path}")
        except IOError as e:
            self.logger.error(f"Error writing to file {output_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error saving features file: {e}")
            raise


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / "data" / "bridge_data.csv"
    output_path = script_dir / "extracted_features.py"

    extractor = FeatureExtractor(str(csv_path))
    extractor.extract_features()
    extractor.display_features()
    extractor.save_features_file(str(output_path))
