from pathlib import Path
import time
import numpy as np

def test_disk_speed():
    """Test raw disk read speed on the training file"""
    file_path = Path("data/openwebtext/train.bin")
    token_size = 2  # bytes per token
    seq_len = 1024  # tokens per sequence
    chunk_size = token_size * seq_len  # ~2KB chunks, matching our actual reads
    
    # Get file size
    file_size = file_path.stat().st_size
    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Chunk size: {chunk_size:,} bytes ({chunk_size/1024:.1f} KB)")
    
    # Sequential read
    print("\nTesting sequential read...")
    start_time = time.time()
    bytes_read = 0
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            
            if bytes_read % (50 * 1024 * 1024) == 0:  # Report every 50MB
                elapsed = time.time() - start_time
                gb_per_second = bytes_read / (1024**3) / elapsed
                print(f"Progress: {bytes_read / file_size * 100:.1f}% - Speed: {gb_per_second:.2f} GB/s")
    
    elapsed = time.time() - start_time
    gb_per_second = file_size / (1024**3) / elapsed
    
    print(f"\nResults:")
    print(f"Total bytes read: {bytes_read:,}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Average speed: {gb_per_second:.2f} GB/second")
    
    # Random read test
    print("\nTesting random reads...")
    num_reads = 100000  # Increased since chunks are smaller
    positions = np.random.randint(0, file_size - chunk_size, num_reads)
    positions.sort()  # Test both sorted and unsorted
    
    for sort_desc in [False, True]:
        if sort_desc:
            positions = positions[::-1]
            print("\nTesting random reads (sorted descending)...")
        else:
            print("\nTesting random reads (sorted ascending)...")
            
        start_time = time.time()
        bytes_read = 0
        
        with open(file_path, 'rb') as f:
            for i, pos in enumerate(positions):
                f.seek(pos)
                chunk = f.read(chunk_size)
                bytes_read += len(chunk)
                
                if (i + 1) % (num_reads // 10) == 0:
                    elapsed = time.time() - start_time
                    gb_per_second = bytes_read / (1024**3) / elapsed
                    print(f"Progress: {(i + 1) / num_reads * 100:.1f}% - Speed: {gb_per_second:.2f} GB/s")
        
        elapsed = time.time() - start_time
        gb_per_second = bytes_read / (1024**3) / elapsed
        
        print(f"\nResults:")
        print(f"Total random reads: {num_reads:,}")
        print(f"Bytes per read: {chunk_size:,}")
        print(f"Total bytes read: {bytes_read:,}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Average speed: {gb_per_second:.2f} GB/second")
    
    # True random read test (no sorting)
    print("\nTesting true random reads (unsorted)...")
    start_time = time.time()
    bytes_read = 0
    
    with open(file_path, 'rb') as f:
        for i, pos in enumerate(np.random.randint(0, file_size - chunk_size, num_reads)):
            f.seek(pos)
            chunk = f.read(chunk_size)
            bytes_read += len(chunk)
            
            if (i + 1) % (num_reads // 10) == 0:
                elapsed = time.time() - start_time
                gb_per_second = bytes_read / (1024**3) / elapsed
                print(f"Progress: {(i + 1) / num_reads * 100:.1f}% - Speed: {gb_per_second:.2f} GB/s")
    
    elapsed = time.time() - start_time
    gb_per_second = bytes_read / (1024**3) / elapsed
    
    print(f"\nResults:")
    print(f"Total random reads: {num_reads:,}")
    print(f"Bytes per read: {chunk_size:,}")
    print(f"Total bytes read: {bytes_read:,}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Average speed: {gb_per_second:.2f} GB/second")

if __name__ == '__main__':
    test_disk_speed() 