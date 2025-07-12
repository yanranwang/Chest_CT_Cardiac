#!/usr/bin/env python3
"""
Data preprocessor - Batch preprocessing for cardiac function training data

This script is used to batch preprocess all training data and save processed data to HDF5 files
to accelerate training process and improve GPU utilization.

Main features:
1. Batch process all image data
2. Save to HDF5 database
3. Generate data index
4. Validate data integrity
5. Support incremental updates

Usage:
    python data_preprocessor.py --config config.json
"""

import os
import h5py
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import hashlib
import multiprocessing as mp
from functools import partial
import logging
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from merlin.data.monai_transforms import ImageTransforms
from merlin.training.cardiac_trainer import load_and_validate_csv_data, build_data_list


class DataPreprocessor:
    """Data preprocessor - Batch preprocessing and saving to HDF5"""
    
    def __init__(self, config: dict):
        self.config = config
        self.setup_logging()
        
        # Output paths
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # HDF5 file paths
        self.hdf5_path = self.output_dir / 'preprocessed_data.h5'
        self.metadata_path = self.output_dir / 'data_metadata.json'
        
        # Preprocessing transforms
        self.transform = ImageTransforms
        
        # Data statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'processing_time': 0
        }
    
    def setup_logging(self):
        """Setup logging"""
        log_file = Path(self.config['output_dir']) / 'preprocessing.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_data_hash(self, data_item: dict) -> str:
        """Generate unique hash for data item"""
        hash_content = f"{data_item['image']}_{data_item['basename']}_{data_item['folder']}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def preprocess_single_item(self, data_item: dict) -> Optional[Tuple[str, dict]]:
        """Preprocess single data item"""
        try:
            # Check if file exists
            image_path = data_item.get('image', '')
            if not os.path.exists(image_path):
                self.logger.warning(f"Image file not found: {image_path}")
                return None
            
            # Generate unique ID
            item_id = self.get_data_hash(data_item)
            
            # Apply transforms
            processed_item = self.transform(data_item)
            
            # Extract image data
            image_tensor = processed_item['image']
            
            # Prepare storage data
            storage_data = {
                'image': image_tensor.numpy(),
                'cardiac_metrics': data_item.get('cardiac_metrics'),
                'patient_id': data_item.get('patient_id'),
                'basename': data_item.get('basename'),
                'folder': data_item.get('folder'),
                'metadata': data_item.get('metadata', {}),
                'original_path': data_item.get('image'),
                'processed_shape': image_tensor.shape,
                'data_type': str(image_tensor.dtype)
            }
            
            return item_id, storage_data
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {data_item.get('image', 'unknown')}, error: {e}")
            return None
    
    def process_batch(self, data_batch: List[dict]) -> List[Tuple[str, dict]]:
        """Process data batch"""
        results = []
        
        for data_item in data_batch:
            result = self.preprocess_single_item(data_item)
            if result:
                results.append(result)
                self.stats['processed_files'] += 1
            else:
                self.stats['failed_files'] += 1
        
        return results
    
    def save_to_hdf5(self, processed_data: List[Tuple[str, dict]]):
        """Save processed data to HDF5 file"""
        self.logger.info(f"Saving {len(processed_data)} processed data items to {self.hdf5_path}")
        
        with h5py.File(self.hdf5_path, 'a') as f:
            # Create groups
            if 'images' not in f:
                f.create_group('images')
            if 'metadata' not in f:
                f.create_group('metadata')
            
            for item_id, data in tqdm(processed_data, desc="Saving to HDF5"):
                # Save image data
                if item_id not in f['images']:
                    f['images'].create_dataset(
                        item_id, 
                        data=data['image'],
                        compression='gzip',
                        compression_opts=9
                    )
                
                # Save metadata
                if item_id not in f['metadata']:
                    # Convert metadata to JSON string for storage
                    metadata_json = json.dumps({
                        'cardiac_metrics': data['cardiac_metrics'].tolist() if data['cardiac_metrics'] is not None else None,
                        'patient_id': data['patient_id'],
                        'basename': data['basename'],
                        'folder': data['folder'],
                        'metadata': data['metadata'],
                        'original_path': data['original_path'],
                        'processed_shape': data['processed_shape'],
                        'data_type': data['data_type']
                    }, ensure_ascii=False)
                    
                    f['metadata'].create_dataset(
                        item_id,
                        data=metadata_json,
                        dtype=h5py.string_dtype(encoding='utf-8')
                    )
    
    def create_index(self, data_list: List[dict]) -> Dict[str, Any]:
        """Create data index"""
        index = {
            'items': {},
            'patient_mapping': {},
            'folder_mapping': {},
            'statistics': {
                'total_items': len(data_list),
                'unique_patients': len(set(item['patient_id'] for item in data_list)),
                'unique_folders': len(set(item['folder'] for item in data_list))
            }
        }
        
        for data_item in data_list:
            item_id = self.get_data_hash(data_item)
            
            # Basic info
            index['items'][item_id] = {
                'patient_id': data_item['patient_id'],
                'basename': data_item['basename'],
                'folder': data_item['folder'],
                'has_cardiac_metrics': data_item.get('cardiac_metrics') is not None,
                'original_path': data_item['image']
            }
            
            # Patient mapping
            patient_id = data_item['patient_id']
            if patient_id not in index['patient_mapping']:
                index['patient_mapping'][patient_id] = []
            index['patient_mapping'][patient_id].append(item_id)
            
            # Folder mapping
            folder = data_item['folder']
            if folder not in index['folder_mapping']:
                index['folder_mapping'][folder] = []
            index['folder_mapping'][folder].append(item_id)
        
        return index
    
    def verify_data_integrity(self) -> bool:
        """Validate data integrity"""
        self.logger.info("Validating data integrity...")
        
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                images_group = f['images']
                metadata_group = f['metadata']
                
                # Check data consistency
                image_keys = set(images_group.keys())
                metadata_keys = set(metadata_group.keys())
                
                if image_keys != metadata_keys:
                    self.logger.error("Image data and metadata keys do not match")
                    return False
                
                # Randomly validate a few samples
                sample_keys = list(image_keys)[:min(10, len(image_keys))]
                
                for key in sample_keys:
                    # Check image data
                    image_data = images_group[key][:]
                    if image_data.size == 0:
                        self.logger.error(f"Image data is empty: {key}")
                        return False
                    
                    # Check metadata
                    metadata_str = metadata_group[key][()]
                    if isinstance(metadata_str, bytes):
                        metadata_str = metadata_str.decode('utf-8')
                    
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        self.logger.error(f"Metadata format error: {key}")
                        return False
                
                self.logger.info(f"Data integrity validation passed, {len(image_keys)} samples")
                return True
                
        except Exception as e:
            self.logger.error(f"Data integrity validation failed: {e}")
            return False
    
    def get_existing_items(self) -> set:
        """Get existing data items"""
        if not self.hdf5_path.exists():
            return set()
        
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'images' in f:
                    return set(f['images'].keys())
                return set()
        except Exception as e:
            self.logger.warning(f"Failed to read existing data: {e}")
            return set()
    
    def preprocess_data(self, data_list: List[dict], force_reprocess: bool = False):
        """Main function for data preprocessing"""
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting preprocessing {len(data_list)} data items")
        self.stats['total_files'] = len(data_list)
        
        # Check existing data
        existing_items = set() if force_reprocess else self.get_existing_items()
        
        # Filter data to be processed
        items_to_process = []
        for data_item in data_list:
            item_id = self.get_data_hash(data_item)
            if item_id not in existing_items:
                items_to_process.append(data_item)
            else:
                self.stats['skipped_files'] += 1
        
        if not items_to_process:
            self.logger.info("All data already preprocessed")
            return
        
        self.logger.info(f"Need to process {len(items_to_process)} new data items")
        
        # Batch processing configuration
        batch_size = self.config.get('preprocess_batch_size', 16)
        num_workers = self.config.get('num_workers', 1)  # Default single process
        enable_multiprocessing = self.config.get('enable_multiprocessing', False)
        
        # Check if multiprocessing is enabled
        if enable_multiprocessing and num_workers > 1:
            self.logger.info(f"Using {num_workers} processes for parallel processing")
            try:
                self._multiprocess_preprocess(items_to_process, batch_size, num_workers)
            except Exception as e:
                self.logger.error(f"Multiprocess processing failed: {e}")
                self.logger.info("Fallback to single process processing...")
                self._single_process_preprocess(items_to_process, batch_size)
        else:
            self.logger.info("Using single process processing")
            self._single_process_preprocess(items_to_process, batch_size)
        
        # Create index
        self.logger.info("Creating data index...")
        index = self.create_index(data_list)
        
        # Save index
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        # Validate data integrity
        if not self.verify_data_integrity():
            raise RuntimeError("Data integrity validation failed")
        
        # Statistics
        self.stats['processing_time'] = time.time() - start_time
        self._print_statistics()
        
        # Save statistics
        with open(self.output_dir / 'preprocessing_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _single_process_preprocess(self, items_to_process: List[dict], batch_size: int):
        """Single process preprocessing"""
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(items_to_process), batch_size), 
                      desc="Preprocessing batch", 
                      total=total_batches,
                      ncols=100):
            batch = items_to_process[i:i + batch_size]
            processed_batch = self.process_batch(batch)
            
            if processed_batch:
                self.save_to_hdf5(processed_batch)
                self.logger.info(f"Processed batch {i//batch_size + 1}/{total_batches}, successfully processed {len(processed_batch)} data items")
            else:
                self.logger.warning(f"Batch {i//batch_size + 1}/{total_batches} processing failed or no valid data")
    
    def _multiprocess_preprocess(self, items_to_process: List[dict], batch_size: int, num_workers: int):
        """Multiprocess preprocessing"""
        # Create data batches
        batches = [items_to_process[i:i + batch_size] for i in range(0, len(items_to_process), batch_size)]
        
        # Use process pool for processing
        with mp.Pool(processes=num_workers) as pool:
            # Use separate processing function to avoid serialization issues
            results = pool.map(process_batch_standalone, batches)
        
        # Save results
        for processed_batch in results:
            if processed_batch:
                self.save_to_hdf5(processed_batch)
    
    def _print_statistics(self):
        """Print statistics"""
        self.logger.info("=" * 80)
        self.logger.info("📊 Preprocessing statistics")
        self.logger.info("=" * 80)
        self.logger.info(f"📁 Total files: {self.stats['total_files']}")
        self.logger.info(f"✅ Successfully processed: {self.stats['processed_files']}")
        self.logger.info(f"⏭️   Skipped files: {self.stats['skipped_files']}")
        self.logger.info(f"❌ Failed files: {self.stats['failed_files']}")
        self.logger.info(f"⏱️   Processing time: {self.stats['processing_time']:.2f} seconds")
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
            self.logger.info(f"📈 Success rate: {success_rate:.2f}%")
        self.logger.info(f"💾 Output files: {self.hdf5_path}")
        self.logger.info(f"📋 Index files: {self.metadata_path}")
        self.logger.info("=" * 80)


def process_batch_standalone(data_batch: List[dict]) -> List[Tuple[str, dict]]:
    """Independent batch processing function, used for multiprocess processing"""
    results = []
    
    # Import necessary modules
    from merlin.data.monai_transforms import ImageTransforms
    import hashlib
    
    transform = ImageTransforms
    
    for data_item in data_batch:
        try:
            # Check if file exists
            image_path = data_item.get('image', '')
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
            
            # Generate unique ID
            hash_content = f"{data_item['image']}_{data_item['basename']}_{data_item['folder']}"
            item_id = hashlib.md5(hash_content.encode()).hexdigest()
            
            # Apply transforms
            processed_item = transform(data_item)
            
            # Extract image data
            image_tensor = processed_item['image']
            
            # Prepare storage data
            storage_data = {
                'image': image_tensor.numpy(),
                'cardiac_metrics': data_item.get('cardiac_metrics'),
                'patient_id': data_item.get('patient_id'),
                'basename': data_item.get('basename'),
                'folder': data_item.get('folder'),
                'metadata': data_item.get('metadata', {}),
                'original_path': data_item.get('image'),
                'processed_shape': image_tensor.shape,
                'data_type': str(image_tensor.dtype)
            }
            
            results.append((item_id, storage_data))
            
        except Exception as e:
            print(f"Preprocessing failed: {data_item.get('image', 'unknown')}, error: {e}")
            continue
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preprocessor')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--force', action='store_true', help='Force reprocess all data')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch processing size')
    parser.add_argument('--num_workers', type=int, help='Number of parallel processing processes')
    parser.add_argument('--enable_multiprocessing', action='store_true', help='Enable multiprocess processing')
    parser.add_argument('--test_mode', type=int, help='Test mode: Process only specified number of data items')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Command line parameter overrides
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['preprocess_batch_size'] = args.batch_size
    if args.num_workers:
        config['num_workers'] = args.num_workers
    if args.enable_multiprocessing:
        config['enable_multiprocessing'] = True
    
    # Default preprocessing configuration
    config.setdefault('preprocess_batch_size', 16)
    config.setdefault('num_workers', 1)  # Default single process
    config.setdefault('enable_multiprocessing', False)  # Default disable multiprocess
    
    try:
        # Load data list
        print("🔍 Loading data...")
        df, cardiac_metric_columns = load_and_validate_csv_data(config)
        data_list = build_data_list(df, config, cardiac_metric_columns)
        
        if not data_list:
            print("❌ No valid data found")
            return
        
        # Test mode: Process only specified number of data
        if args.test_mode:
            original_size = len(data_list)
            data_list = data_list[:args.test_mode]
            print(f"🧪 Test mode: Select first {len(data_list)} from {original_size} data items")
        
        # Create preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Start preprocessing
        print("🚀 Starting data preprocessing...")
        preprocessor.preprocess_data(data_list, force_reprocess=args.force)
        
        print("🎉 Data preprocessing completed!")
        print(f"📁 Preprocessed data saved in: {preprocessor.hdf5_path}")
        print(f"📋 Data index saved in: {preprocessor.metadata_path}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    main() 