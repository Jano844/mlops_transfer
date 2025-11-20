import pandas as pd
import os
from datetime import datetime

class DataConsolidator:
    def __init__(self, raw_data_dir='raw_data/', output_dir='cleaned_data/'):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.forklifts = [
            '5a446af4.csv', '8ec4934b.csv', '8ff9320e.csv', '13e4e55d.csv',
            '63ec02d6.csv', '98c3804e.csv', '713ffecd.csv', '06394b06.csv',
            '39559d5a.csv', '2993457c.csv', 'a3315a74.csv', 'd6d5bfa3.csv',
            'd88bfaa3.csv', 'd3446ef6.csv', 'f092e16f.csv', 'ffd43497.csv'
        ]
        self.tow_trucks = [
            '46f2a9fe.csv', '541435ba.csv', 'f28ba89e.csv', 'c0e45e36.csv'
        ]
        self.header = ['Identifier', 'Height', 'Loaded', 'Onduty', 'TimeStamp', 
                      'Latitude', 'Longtitude', 'Speed']
    def clean_and_filter_data(self, file_path):
        """
        Clean and filter data for a single file, keeping only:
        - All on-duty rows (Onduty=1)
        - One row before turning on (transition from 0 to 1)
        - One row after turning off (transition from 1 to 0)
        Also remove rows with empty/missing values
        """
        try:
            df = pd.read_csv(file_path, sep=';', header=None, names=self.header)
            df = df[df.notna().sum(axis=1) >= 5]
            df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
            df['Loaded'] = pd.to_numeric(df['Loaded'], errors='coerce').fillna(0).astype(int)
            df['Onduty'] = pd.to_numeric(df['Onduty'], errors='coerce').fillna(0).astype(int)
            df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longtitude'] = pd.to_numeric(df['Longtitude'], errors='coerce')
            df['Speed'] = pd.to_numeric(df['Speed'], errors='coerce')
            df = df[
                (df['Latitude'].between(-90, 90)) & 
                (df['Longtitude'].between(-180, 180)) &
                (df['TimeStamp'] > 0) &  # Valid timestamp
                (df['Onduty'].notna())   # Valid Onduty value
            ]
            df = df.sort_values('TimeStamp').reset_index(drop=True)
            if len(df) == 0:
                return pd.DataFrame()
            df['Onduty_next'] = df['Onduty'].shift(-1).fillna(df['Onduty'])
            df['Onduty_prev'] = df['Onduty'].shift(1).fillna(df['Onduty'])
            keep_mask = (
                (df['Onduty'] == 1) |
                ((df['Onduty'] == 0) & (df['Onduty_next'] == 1)) |
                ((df['Onduty'] == 0) & (df['Onduty_prev'] == 1))
            )
            filtered_df = df[keep_mask].copy()
            filtered_df = filtered_df[self.header]
            print(f"Processed {file_path}: {len(df)} -> {len(filtered_df)} rows "
                  f"({len(filtered_df)/len(df)*100:.1f}% kept)")
            return filtered_df
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return pd.DataFrame()
    
    def find_on_duty_periods(self, file_path):
        try:
            df = pd.read_csv(file_path, sep=';', header=None, names=self.header)
            df = df[df.notna().sum(axis=1) >= 5]
            df['Onduty'] = pd.to_numeric(df['Onduty'], errors='coerce').fillna(0).astype(int)
            df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')
            df = df.sort_values('TimeStamp').reset_index(drop=True)
            if len(df) == 0:
                return pd.DataFrame()
            df['Onduty_change'] = df['Onduty'].diff().fillna(1)
            on_starts = df[(df['Onduty'] == 1) & (df['Onduty_change'] == 1)].index
            on_ends = df[(df['Onduty'] == 0) & (df['Onduty_change'] == -1)].index
            keep_indices = set()
            for start_idx in on_starts:
                if start_idx > 0:
                    keep_indices.add(start_idx - 1)  # The off->on transition
                keep_indices.add(start_idx)  # First on-duty row
            for end_idx in on_ends:
                keep_indices.add(end_idx)  # The on->off transition
                if end_idx < len(df) - 1:
                    keep_indices.add(end_idx + 1)  # First off-duty row after
            on_duty_indices = df[df['Onduty'] == 1].index
            keep_indices.update(on_duty_indices)
            keep_indices = sorted(keep_indices)
            filtered_df = df.iloc[keep_indices].copy()
            filtered_df = filtered_df[self.header]
            print(f"Processed {file_path}: {len(df)} -> {len(filtered_df)} rows "
                  f"({len(filtered_df)/len(df)*100:.1f}% kept)")
            return filtered_df
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return pd.DataFrame()
    
    def process_vehicle_category(self, file_list, category_name):
        print(f"\nProcessing {category_name}...")
        print("=" * 50)
        all_vehicles_data = []
        processed_count = 0
        for filename in file_list:
            file_path = os.path.join(self.raw_data_dir, filename)
            if os.path.exists(file_path):
                vehicle_data = self.find_on_duty_periods(file_path)
                if not vehicle_data.empty:
                    all_vehicles_data.append(vehicle_data)
                    processed_count += 1
            else:
                print(f"File not found: {file_path}")
        
        if all_vehicles_data:
            combined_data = pd.concat(all_vehicles_data, ignore_index=True)
            combined_data = combined_data.sort_values('TimeStamp').reset_index(drop=True)
            output_file = os.path.join(self.output_dir, f"{category_name.lower()}_combined.csv")
            combined_data.to_csv(output_file, sep=';', index=False)
            print(f"\n{category_name} Summary:")
            print(f"  Processed {processed_count} files")
            print(f"  Total rows: {len(combined_data):,}")
            if len(combined_data) > 0:
                onduty_percentage = (combined_data['Onduty'] == 1).mean() * 100
                print(f"  On-duty rows: {onduty_percentage:.1f}%")
            if len(combined_data) > 0:
                time_range = f"{self.format_timestamp(combined_data['TimeStamp'].min())} to {self.format_timestamp(combined_data['TimeStamp'].max())}"
                print(f"  Time range: {time_range}")
            print(f"  Saved to: {output_file}")
            return combined_data
        else:
            print(f"No data found for {category_name}")
            return pd.DataFrame()
    def format_timestamp(self, timestamp_ms):
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return "Invalid timestamp"
    def run_consolidation_pipeline(self):
        print("Starting Data Consolidation Pipeline")
        print("=" * 60)
        forklift_data = self.process_vehicle_category(self.forklifts, "Forklifts")
        tow_truck_data = self.process_vehicle_category(self.tow_trucks, "TowTrucks")
        print("\n" + "=" * 60)
        print("Data Consolidation Completed Successfully!")
        print(f"Output files saved to: {self.output_dir}")
def main():
    consolidator = DataConsolidator(
        raw_data_dir='raw_data/',
        output_dir='cleaned_data/'
    )
    consolidator.run_consolidation_pipeline()
if __name__ == "__main__":
    main()
