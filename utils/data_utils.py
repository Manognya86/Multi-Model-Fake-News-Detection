import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not text or pd.isna(text):
        return ""
    
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        if 'title' not in df.columns:
            text_col = df.columns[0]
            df['title'] = df[text_col].astype(str)
        
        df['clean_text'] = df['title'].apply(lambda x: clean_text(x) if isinstance(x, str) else "")
        
        if 'tweet_ids' in df.columns:
            df['tweet_ids'] = df['tweet_ids'].apply(lambda x: process_tweet_ids(x) if pd.notna(x) else [])
            df['tweet_count'] = df['tweet_ids'].apply(len)
        else:
            df['tweet_count'] = 0
        
        df['text_length'] = df['clean_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        return df
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame({
            'title': [],
            'clean_text': [],
            'tweet_count': [],
            'text_length': []
        })

def process_tweet_ids(tweet_ids_str):
    if pd.isna(tweet_ids_str):
        return []
    
    try:
        tweet_ids = re.split(r'[\t\s]+', str(tweet_ids_str))
        tweet_ids = [tid for tid in tweet_ids if tid]
        return tweet_ids
    except:
        return []

def prepare_datasets(data_dir, dataset_files, test_size=0.2, val_size=0.1):
    all_dfs = []
    
    for file_name in dataset_files:
        file_path = os.path.join(data_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                df = load_and_preprocess_data(file_path)
                
                if not df.empty:
                    if 'fake' in file_name.lower():
                        df['label'] = 0
                    else:
                        df['label'] = 1
                        
                    all_dfs.append(df)
                    print(f"Loaded {len(df)} entries from {file_name}")
                else:
                    print(f"Empty dataframe for {file_name}")
                    
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    if not all_dfs:
        print("No dataset files found. Creating sample data...")
        sample_data = {
            'title': [f'Sample news title {i}' for i in range(100)],
            'tweet_ids': [f"{i} {i+1} {i+2}" for i in range(100)],
            'label': [i % 2 for i in range(100)],
        }
        combined_df = pd.DataFrame(sample_data)
        combined_df = load_and_preprocess_data(combined_df)
    else:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    
    train_df, test_df = train_test_split(
        combined_df, 
        test_size=test_size, 
        random_state=42, 
        stratify=combined_df['label']
    )
    
    train_df, val_df = train_test_split(
        train_df, 
        test_size=val_size, 
        random_state=42, 
        stratify=train_df['label']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df