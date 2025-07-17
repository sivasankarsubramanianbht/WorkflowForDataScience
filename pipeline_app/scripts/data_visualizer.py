import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

class DataVisualizer:
    """
    Handles the generation and saving of various plots for the flight delay dataset.
    """
    def __init__(self, output_dir="plots"):
        """
        Initializes the DataVisualizer.

        Args:
            output_dir (str): The directory where the plots will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) # Ensure the output directory exists

    def plot_column_distribution(self, df: pd.DataFrame, n_graph_shown: int = 10, n_graph_per_row: int = 4, filename: str = "column_distributions.png"):
        """
        Generates distribution graphs (histogram/bar graph) for columns with
        2-49 unique values and saves the plot.

        Args:
            df (pd.DataFrame): The input DataFrame.
            n_graph_shown (int): Maximum number of graphs to show.
            n_graph_per_row (int): Number of graphs per row in the plot.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating column distribution plots and saving to {filename}...")
        nunique = df.nunique()
        # Filter columns to only include those with 2 to 49 unique values (categorical/low-cardinality numerical)
        df_filtered = df[[col for col in df.columns if nunique[col] > 1 and nunique[col] < 50]]
        
        n_col = df_filtered.shape[1]
        if n_col == 0:
            print("No suitable columns found for distribution plotting (2-49 unique values).")
            return

        column_names = list(df_filtered.columns)
        n_graph_row = math.ceil(min(n_col, n_graph_shown) / n_graph_per_row)

        plt.figure(num=None, figsize=(6 * n_graph_per_row, 5 * n_graph_row), dpi=80, facecolor='w', edgecolor='k')
        plt.suptitle("Distribution of Selected Columns", fontsize=16, y=1.02) # Add a main title

        for i in range(min(n_col, n_graph_shown)):
            plt.subplot(n_graph_row, n_graph_per_row, i + 1)
            column_df = df_filtered.iloc[:, i]
            
            # Check if the column is numerical or object/category
            if pd.api.types.is_numeric_dtype(column_df):
                column_df.hist(bins=20) # Use more bins for histograms
                plt.ylabel('Frequency')
            else:
                value_counts = column_df.value_counts()
                value_counts.plot.bar()
                plt.ylabel('Counts')
            
            plt.xticks(rotation=90)
            plt.title(f'{column_names[i]}')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close() # Close the plot to free memory

    def plot_airline_counts(self, df: pd.DataFrame, filename: str = "airline_counts.png"):
        """
        Generates a bar plot showing the occurrence of each airline and saves it.

        Args:
            df (pd.DataFrame): The input DataFrame.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating airline counts plot and saving to {filename}...")
        df_category = df.select_dtypes(include=['object', 'category'])
        if 'AIRLINE' not in df_category.columns:
            print("AIRLINE column not found for plotting.")
            return

        order_desc = df_category['AIRLINE'].value_counts().index
        plt.figure(figsize=(10, len(order_desc) * 0.5 + 1)) # Dynamic figure size
        sns.set_context("notebook")
        g = sns.countplot(y="AIRLINE", data=df_category, order=order_desc, palette='viridis')
        g.set(xlabel="Number of flights", title="Flight Counts by Airline")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_destination_visits(self, df: pd.DataFrame, top_n: int = 20, filename: str = "top_destination_visits.png"):
        """
        Generates a dot plot showing the number of visits to each destination airport,
        focusing on the top destinations, and saves the plot.

        Args:
            df (pd.DataFrame): The input DataFrame.
            top_n (int): Number of top destinations to plot.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating top {top_n} destination visits plot and saving to {filename}...")
        if 'DEST' not in df.columns:
            print("DEST column not found for plotting.")
            return

        dest_counts = df['DEST'].value_counts().reset_index()
        dest_counts.columns = ['DEST', 'count']
        dest_counts = dest_counts.head(top_n).sort_values('count')

        plt.figure(figsize=(12, 8))
        colors = sns.color_palette('viridis', len(dest_counts))

        plt.scatter(
            dest_counts['count'],
            dest_counts['DEST'],
            s=150,
            color=colors,
            edgecolors='black',
            alpha=0.8
        )

        for i, (count, dest) in enumerate(zip(dest_counts['count'], dest_counts['DEST'])):
            plt.text(
                count + max(dest_counts['count']) * 0.01,
                dest,
                f'{count:,}', # Format with comma for thousands
                va='center',
                ha='left',
                fontsize=10
            )

        plt.title(f'Top {top_n} Destination Airports by Number of Visits', fontsize=16, pad=15)
        plt.xlabel('Number of Visits', fontsize=14)
        plt.ylabel('Destination Airport', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.4)

        min_count = dest_counts['count'].min()
        max_count = dest_counts['count'].max()
        padding = (max_count - min_count) * 0.1
        plt.xlim(min_count - padding, max_count + padding)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_average_arrival_delay_by_airline(self, df: pd.DataFrame, min_flight_count: int = 100, filename: str = "avg_arrival_delay_by_airline.png"):
        """
        Generates a dot plot showing average arrival delays by airline for delayed flights,
        and saves the plot.

        Args:
            df (pd.DataFrame): The input DataFrame.
            min_flight_count (int): Minimum number of flights an airline must have to be included.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating average arrival delay by airline plot and saving to {filename}...")
        
        # Clean data - remove rows with missing airline or delay info
        df_cleaned = df.dropna(subset=['AIRLINE', 'ARR_DELAY'])
        
        # Filter for delayed flights only
        df_delayed = df_cleaned[df_cleaned['ARR_DELAY'] > 0]
        
        if df_delayed.empty:
            print("No delayed flights found to plot average delays by airline.")
            return

        # Calculate average delays by airline
        avg_delay = df_delayed.groupby('AIRLINE')['ARR_DELAY'].agg(['mean', 'count']).reset_index()
        avg_delay = avg_delay.rename(columns={'mean': 'avg_delay', 'count': 'flight_count'})
        
        # Filter airlines with sufficient data
        avg_delay = avg_delay[avg_delay['flight_count'] >= min_flight_count].sort_values('avg_delay').reset_index(drop=True)
        
        if avg_delay.empty:
            print(f"No airlines found with at least {min_flight_count} delayed flights to plot.")
            return

        plt.figure(figsize=(12, max(8, len(avg_delay) * 0.6))) # Adjust figure height dynamically
        colors = sns.color_palette('viridis', len(avg_delay))

        plt.scatter(
            avg_delay['avg_delay'],
            avg_delay['AIRLINE'],
            s=150,
            color=colors,
            edgecolors='black',
            alpha=0.8
        )

        for i, (delay, count) in enumerate(zip(avg_delay['avg_delay'], avg_delay['flight_count'])):
            plt.text(
                delay + 0.5, i,
                f'{delay:.1f} min\n({count:,} flights)',
                va='center',
                ha='left',
                fontsize=10
            )

        plt.title('Average Arrival Delay by Airline (Flights with Delays Only)', fontsize=16, pad=20)
        plt.xlabel('Average Delay Duration (minutes)', fontsize=14)
        plt.ylabel('Airline Code', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.4)

        min_delay = avg_delay['avg_delay'].min()
        max_delay = avg_delay['avg_delay'].max()
        padding = (max_delay - min_delay) * 0.1
        plt.xlim(min_delay - padding, max_delay + padding)

        mean_delay = df_delayed['ARR_DELAY'].mean()
        plt.axvline(mean_delay, color='red', linestyle='--', alpha=0.7, label=f'Overall Average: {mean_delay:.1f} min')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        print("\nAdditional Insights (Average Delay by Airline):")
        print(f"- Worst performing airline: {avg_delay.iloc[-1]['AIRLINE']} ({avg_delay.iloc[-1]['avg_delay']:.1f} min avg delay)")
        print(f"- Best performing airline: {avg_delay.iloc[0]['AIRLINE']} ({avg_delay.iloc[0]['avg_delay']:.1f} min avg delay)")
        print(f"- Total flights analyzed for this plot: {avg_delay['flight_count'].sum():,}")
        print(f"- Percentage of flights considered 'delayed' for this analysis: {len(df_delayed) / len(df_cleaned) * 100:.1f}%")

    def plot_total_delays_by_year(self, df: pd.DataFrame, filename: str = "total_delays_by_year.png"):
        """
        Generates a bar plot showing total arrival delays by year and saves it.

        Args:
            df (pd.DataFrame): The input DataFrame.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating total delays by year plot and saving to {filename}...")
        
        # Ensure 'FL_DATE' is datetime and 'ARR_DELAY' exists
        if 'FL_DATE' not in df.columns or 'ARR_DELAY' not in df.columns:
            print("Required columns 'FL_DATE' or 'ARR_DELAY' not found for plotting total delays by year.")
            return

        df['YEAR'] = pd.to_datetime(df['FL_DATE']).dt.year
        yearly_delays = df.groupby('YEAR')['ARR_DELAY'].sum().reset_index()

        if yearly_delays.empty:
            print("No data to plot total delays by year.")
            return

        max_delay_year = yearly_delays.loc[yearly_delays['ARR_DELAY'].idxmax(), 'YEAR']
        max_delay_value = yearly_delays['ARR_DELAY'].max()

        plt.figure(figsize=(12, 6))
        barplot = sns.barplot(
            x='YEAR',
            y='ARR_DELAY',
            data=yearly_delays,
            palette='viridis'
        )

        # Highlight the year with most delays
        if max_delay_year in yearly_delays['YEAR'].values:
            idx = yearly_delays[yearly_delays['YEAR'] == max_delay_year].index[0]
            barplot.patches[idx].set_facecolor('red')

        plt.title(f'Total Flight Delays by Year (Max: {max_delay_year} with {max_delay_value:,.0f} min)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Total Delay Minutes', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for p in barplot.patches:
            barplot.annotate(
                f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points'
            )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"The year with the most delays was {max_delay_year} with {max_delay_value:,.0f} total delay minutes.")


    def plot_monthly_delays_by_year(self, df: pd.DataFrame, filename: str = "monthly_delays_by_year.png"):
        """
        Generates a line plot showing monthly total arrival delay minutes by year and saves it.

        Args:
            df (pd.DataFrame): The input DataFrame.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating monthly delays by year plot and saving to {filename}...")
        
        # Ensure 'FL_DATE' is datetime and 'ARR_DELAY' exists
        if 'FL_DATE' not in df.columns or 'ARR_DELAY' not in df.columns:
            print("Required columns 'FL_DATE' or 'ARR_DELAY' not found for plotting monthly delays by year.")
            return

        df['MONTH'] = pd.to_datetime(df['FL_DATE']).dt.month
        df['YEAR'] = pd.to_datetime(df['FL_DATE']).dt.year # Ensure YEAR is also present
        monthly_delays = df.groupby(['YEAR', 'MONTH'])['ARR_DELAY'].sum().reset_index()

        if monthly_delays.empty:
            print("No data to plot monthly delays by year.")
            return

        plt.figure(figsize=(14, 7))
        sns.lineplot(
            x='MONTH',
            y='ARR_DELAY',
            hue='YEAR',
            data=monthly_delays,
            palette='tab10',
            marker='o',
            linewidth=2.5
        )

        plt.title('Monthly Flight Delays by Year', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Delay Minutes', fontsize=12)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_monthly_trend_with_highlight(self, df: pd.DataFrame, column: str, title: str, ylabel: str, filename: str = "monthly_trend_highlight.png"):
        """
        Generates a line plot showing monthly total delays over time, highlighting the year
        with the least delays, and saves the plot.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column containing delay values (e.g., 'ARR_DELAY').
            title (str): The main title for the plot.
            ylabel (str): The label for the y-axis.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating monthly trend with highlight plot and saving to {filename}...")

        # Ensure required columns exist
        if 'FL_DATE' not in df.columns or column not in df.columns:
            print(f"Required columns 'FL_DATE' or '{column}' not found for plotting monthly trend.")
            return
            
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        df_copy['year'] = pd.to_datetime(df_copy['FL_DATE']).dt.year
        df_copy['month'] = pd.to_datetime(df_copy['FL_DATE']).dt.month
        
        yearly_sums = df_copy.groupby('year')[column].sum().reset_index()
        if yearly_sums.empty:
            print("No yearly data to plot monthly trend.")
            return

        min_val_year = yearly_sums.loc[yearly_sums[column].idxmin(), 'year']
        min_val_value = yearly_sums[column].min()
        
        trend_data = df_copy.groupby(['year', 'month'])[column].sum().reset_index()
        trend_data['date'] = pd.to_datetime(trend_data[['year', 'month']].assign(day=1))
        
        if trend_data.empty:
            print("No monthly trend data to plot.")
            return

        plt.figure(figsize=(14, 6))
        
        # Plot all data in light gray
        sns.lineplot(x='date', y=column, data=trend_data, 
                     marker='o', color='lightgray', alpha=0.5, label='Other years', legend=False) # No legend for overall
        
        # Highlight the minimum year in orange
        min_year_data = trend_data[trend_data['year'] == min_val_year]
        sns.lineplot(x='date', y=column, data=min_year_data, 
                     marker='o', color='orange', linewidth=2, label=f'Min Year: {min_val_year}', legend='full') # Legend for highlighted

        plt.title(f'{title}\nYear with Least {ylabel.split(" ")[-1]}: {min_val_year} ({min_val_value:,.0f})')
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        print(f"The year with the least {ylabel.split(' ')[-1].lower()} was {min_val_year} with {min_val_value:,.0f} total.")
        return min_val_year

    def plot_delay_reason_analysis(self, df: pd.DataFrame, filename: str = "delay_reason_analysis.png"):
        """
        Analyzes and plots the distribution of delay reasons using pie and bar charts.
        Assumes the input df already contains the individual delay reason columns.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing delay reason columns.
            filename (str): Name of the file to save the plot.
        """
        print(f"Generating delay reason analysis plots and saving to {filename}...")

        delay_reasons = [
            'DELAY_DUE_CARRIER',
            'DELAY_DUE_WEATHER',
            'DELAY_DUE_NAS',
            'DELAY_DUE_SECURITY',
            'DELAY_DUE_LATE_AIRCRAFT'
        ]
        
        # Check if all delay reason columns exist
        missing_cols = [col for col in delay_reasons if col not in df.columns]
        if missing_cols:
            print(f"Missing delay reason columns: {missing_cols}. Skipping delay reason analysis.")
            return

        # Calculate total delay minutes by reason
        total_delays = df[delay_reasons].sum().sort_values(ascending=False)
        
        if total_delays.sum() == 0:
            print("No recorded delays in the specified reason columns to analyze.")
            return

        # Calculate percentage contribution
        delay_percentages = (total_delays / total_delays.sum()) * 100

        plt.figure(figsize=(15, 8))

        # Pie chart showing proportion of delay reasons
        plt.subplot(1, 2, 1)
        plt.pie(delay_percentages,
                labels=[f"{label.replace('DELAY_DUE_', '').replace('_', ' ').title()}\n{percent:.1f}%"
                        for label, percent in zip(delay_percentages.index, delay_percentages)],
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette('pastel'))
        plt.title('Proportion of Delay Reasons')

        # Bar plot showing absolute delay minutes
        plt.subplot(1, 2, 2)
        ax = sns.barplot(x=total_delays.values, y=total_delays.index,
                        palette='viridis')
        plt.title('Total Delay Minutes by Reason')
        plt.xlabel('Total Delay Minutes')
        plt.ylabel('Delay Reason')

        # Add value labels to bars
        for i, v in enumerate(total_delays):
            ax.text(v + max(total_delays)*0.01, i,
                   f"{v:,.0f} min",
                   color='black', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

        print("\nDelay Reason Analysis Summary:")
        print("------------------------------")
        for reason, minutes in total_delays.items():
            clean_name = reason.replace('DELAY_DUE_', '').replace('_', ' ').title()
            print(f"{clean_name:<20}: {minutes:>12,.0f} minutes ({delay_percentages[reason]:.1f}%)")
        
        primary_reason = total_delays.idxmax()
        clean_primary = primary_reason.replace('DELAY_DUE_', '').replace('_', ' ').title()
        print(f"\nThe primary reason for delays is: {clean_primary} "
              f"({delay_percentages[primary_reason]:.1f}% of all delays)")