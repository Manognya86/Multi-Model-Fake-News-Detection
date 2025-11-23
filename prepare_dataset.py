import pandas as pd
import os
import shutil
from tqdm import tqdm
import glob
import re

def prepare_datasets():
    """
    Prepare datasets from all available sources including Twitter15/16 with tree data
    """
    data_path = "data"
    
    # Create processed data directory
    processed_path = os.path.join(data_path, "processed")
    os.makedirs(processed_path, exist_ok=True)
    
    all_datasets = []
    
    # Process CSV files
    csv_files = [
        "gossipcop_fake.csv", "gossipcop_real.csv",
        "politifact_fake.csv", "politifact_real.csv", 
        "snopes_medical.csv", "data1.csv", "data2.csv", "data3.csv"
    ]
    
    for csv_file in csv_files:
        file_path = os.path.join(data_path, csv_file)
        
        if os.path.exists(file_path):
            print(f"Processing {csv_file}")
            
            try:
                df = pd.read_csv(file_path)
                
                # Add dataset and label columns
                dataset_name = csv_file.replace('.csv', '')
                if 'fake' in csv_file.lower():
                    df['label'] = 1  # Fake news
                elif 'real' in csv_file.lower():
                    df['label'] = 0  # Real news
                else:
                    df['label'] = 0  # Default to real
                
                df['dataset'] = dataset_name
                df['has_tree'] = False  # CSV files don't have tree data
                
                # Create post_id if not exists
                if 'post_id' not in df.columns:
                    df['post_id'] = [f"{dataset_name}_{i}" for i in range(len(df))]
                
                # Add image references
                df = add_image_references(df, data_path, dataset_name)
                
                all_datasets.append(df)
                print(f"  Processed {len(df)} entries from {csv_file}")
                
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
    
    # Process Twitter15 and Twitter16 datasets
    twitter_datasets = ['twitter15', 'twitter16']
    for twitter_dataset in twitter_datasets:
        twitter_path = os.path.join(data_path, twitter_dataset)
        if os.path.exists(twitter_path):
            print(f"Processing {twitter_dataset}")
            twitter_df = process_twitter_dataset(twitter_path, twitter_dataset)
            all_datasets.append(twitter_df)
            print(f"  Processed {len(twitter_df)} entries from {twitter_dataset}")
    
    if all_datasets:
        # Combine all datasets
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Fill missing columns
        combined_df = fill_missing_columns(combined_df)
        
        # Split into train, validation, and test sets
        train_df = combined_df.sample(frac=0.7, random_state=42)
        temp_df = combined_df.drop(train_df.index)
        val_df = temp_df.sample(frac=0.15, random_state=42)
        test_df = temp_df.drop(val_df.index)
        
        # Save splits
        train_df.to_csv(os.path.join(processed_path, "train_data.csv"), index=False)
        val_df.to_csv(os.path.join(processed_path, "val_data.csv"), index=False)
        test_df.to_csv(os.path.join(processed_path, "test_data.csv"), index=False)
        
        # Print dataset statistics
        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Total entries: {len(combined_df)}")
        print(f"   Training samples: {len(train_df)}")
        print(f"   Validation samples: {len(val_df)}")
        print(f"   Test samples: {len(test_df)}")
        print(f"   Fake news: {len(combined_df[combined_df['label'] == 1])}")
        print(f"   Real news: {len(combined_df[combined_df['label'] == 0])}")
        print(f"   Entries with tree data: {combined_df['has_tree'].sum()}")
        print(f"   Entries with images: {combined_df['has_image'].sum()}")
        
        # Dataset sources
        print(f"\n📁 Dataset Sources:")
        for dataset in combined_df['dataset'].unique():
            count = len(combined_df[combined_df['dataset'] == dataset])
            fake_count = len(combined_df[(combined_df['dataset'] == dataset) & (combined_df['label'] == 1)])
            real_count = len(combined_df[(combined_df['dataset'] == dataset) & (combined_df['label'] == 0)])
            print(f"   {dataset}: {count} total ({fake_count} fake, {real_count} real)")
    
    else:
        print("No datasets found! Creating sample data...")
        create_sample_data(processed_path)
    
    print("✅ Dataset preparation complete!")

def process_twitter_dataset(twitter_path, dataset_name):
    """Process Twitter15 or Twitter16 dataset with tree data"""
    fake_path = os.path.join(twitter_path, "fake")
    real_path = os.path.join(twitter_path, "real")
    
    data = []
    
    # Process fake samples
    if os.path.exists(fake_path):
        for file in os.listdir(fake_path):
            if file.endswith('.txt'):
                sample_id = extract_sample_id(file)
                file_path = os.path.join(fake_path, file)
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                data.append({
                    'post_id': f"{dataset_name}_fake_{sample_id}",
                    'text': content,
                    'title': f"Twitter {dataset_name} Fake Sample {sample_id}",
                    'label': 1,  # Fake
                    'dataset': dataset_name,
                    'has_tree': True,
                    'sample_id': sample_id,
                    'tweet_count': 10,  # Default value
                    'text_length': len(content)
                })
    
    # Process real samples
    if os.path.exists(real_path):
        for file in os.listdir(real_path):
            if file.endswith('.txt'):
                sample_id = extract_sample_id(file)
                file_path = os.path.join(real_path, file)
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                data.append({
                    'post_id': f"{dataset_name}_real_{sample_id}",
                    'text': content,
                    'title': f"Twitter {dataset_name} Real Sample {sample_id}",
                    'label': 0,  # Real
                    'dataset': dataset_name,
                    'has_tree': True,
                    'sample_id': sample_id,
                    'tweet_count': 10,  # Default value
                    'text_length': len(content)
                })
    
    return pd.DataFrame(data)

def extract_sample_id(filename):
    """Extract sample ID from filename"""
    match = re.search(r'sample_(\d+)', filename)
    if match:
        return match.group(1)
    return "0"

def add_image_references(df, data_path, dataset_name):
    """Add image references to dataframe"""
    image_path = os.path.join(data_path, "images")
    available_images = set()
    
    if os.path.exists(image_path):
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            available_images.update([os.path.basename(f) for f in glob.glob(os.path.join(image_path, ext))])
    
    df['image_path'] = None
    df['has_image'] = False
    
    for idx, row in df.iterrows():
        post_id = row['post_id']
        
        # Try different naming conventions for images
        possible_names = [
            f"{post_id}.jpg",
            f"{post_id}.jpeg", 
            f"{post_id}.png",
            f"{dataset_name}_{idx}.jpg",
        ]
        
        for img_name in possible_names:
            if img_name in available_images:
                df.at[idx, 'image_path'] = os.path.join(image_path, img_name)
                df.at[idx, 'has_image'] = True
                break
    
    return df

def fill_missing_columns(df):
    """Fill missing columns that are required for processing"""
    required_columns = {
        'text': '',
        'title': '', 
        'tweet_count': 10,
        'text_length': 0,
        'user_followers': 100,
        'user_friends': 50,
        'retweet_count': 5,
        'favorite_count': 3,
        'has_tree': False,
        'sample_id': '0'
    }
    
    for col, default_value in required_columns.items():
        if col not in df.columns:
            if col == 'text' and 'title' in df.columns:
                df['text'] = df['title']
            elif col == 'title' and 'text' in df.columns:
                df['title'] = df['text']
            else:
                df[col] = default_value
    
    # Ensure text column is string type
    df['text'] = df['text'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    
    # Calculate text length
    df['text_length'] = df['text'].apply(len)
    
    return df

def create_sample_data(processed_path):
    """Create sample data if no datasets are found"""
    sample_data = {
        'post_id': [f'sample_{i}' for i in range(100)],
        'text': [f'Sample text content for post {i}' for i in range(100)],
        'title': [f'Sample title {i}' for i in range(100)],
        'label': [i % 2 for i in range(100)],
        'tweet_count': [i % 20 for i in range(100)],
        'text_length': [len(f'Sample text content for post {i}') for i in range(100)],
        'dataset': ['sample'] * 100,
        'has_tree': [False] * 100,
        'sample_id': [str(i) for i in range(100)],
        'image_path': [None] * 100,
        'has_image': [False] * 100
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Split
    train_df = sample_df.sample(frac=0.7, random_state=42)
    temp_df = sample_df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.15, random_state=42)
    test_df = temp_df.drop(val_df.index)
    
    # Save
    train_df.to_csv(os.path.join(processed_path, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(processed_path, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(processed_path, "test_data.csv"), index=False)
    
    print("Created sample data with 100 entries")

if __name__ == '__main__':
    prepare_datasets()