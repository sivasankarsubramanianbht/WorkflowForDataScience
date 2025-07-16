import pandas as pd
from ydata_profiling import ProfileReport
import os

class DataProfiler:
    """
    Handles the generation of a comprehensive profiling report for a given DataFrame.
    """
    def __init__(self, output_dir="reports"):
        """
        Initializes the DataProfiler.

        Args:
            output_dir (str): The directory where the profiling report will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Ensure the output directory exists

    def generate_profile_report(self, df: pd.DataFrame, report_name: str = "profile_report.html"):
        """
        Generates a Pandas Profiling report for the given DataFrame and saves it
        to an HTML file.

        Args:
            df (pd.DataFrame): The DataFrame for which to generate the report.
            report_name (str): The name of the output HTML file for the report.
                               Defaults to "profile_report.html".
        Returns:
            str: The full path to the generated report file.
        """
        print(f"Generating profile report: {report_name}...")
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        
        output_path = os.path.join(self.output_dir, report_name)
        profile.to_file(output_path)
        print(f"Profile report saved to: {output_path}")
        return output_path

# This file itself does not execute the profiling.
# It provides the DataProfiler class to be imported and used by main.py or other scripts.