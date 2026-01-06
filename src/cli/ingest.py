"""
Data Pipeline CLI - Load products into Milvus vector database.

Usage:
    uv run python -m src.cli.ingest --sync      # Smart sync (auto-detect)
    uv run python -m src.cli.ingest --full      # Force full reload
    uv run python -m src.cli.ingest --stats     # Check database stats
"""

import argparse
import sys
import warnings

# Suppress pkg_resources deprecation warning from milvus_lite
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from src.config import settings


def get_product_count() -> int:
    """Get current product count in database."""
    from pymilvus import MilvusClient
    
    try:
        client = MilvusClient(settings.milvus_db_path)
        if "products" in client.list_collections():
            stats = client.get_collection_stats("products")
            count = stats.get("row_count", 0)
            client.close()
            return count
        client.close()
    except Exception:
        pass
    return 0


def show_stats():
    """Show database statistics."""
    from pymilvus import MilvusClient
    
    print("Connecting to Milvus...")
    client = MilvusClient(settings.milvus_db_path)
    
    collections = client.list_collections()
    print(f"Collections: {collections}")
    
    if "products" in collections:
        stats = client.get_collection_stats("products")
        print(f"Products count: {stats['row_count']}")
        
        # Show sample data
        try:
            results = client.query(
                collection_name="products",
                output_fields=["product_id", "name", "category", "price"],
                limit=5
            )
            print(f"\nSample products ({len(results)}):")
            print("-" * 60)
            for r in results:
                print(f"  {r['product_id']}: {r['name'][:40]}... ${r['price']:.2f}")
        except Exception:
            pass
    
    client.close()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Load product data into Milvus vector database"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Smart sync: full reload if DB empty, otherwise incremental update",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full reload: drop existing data and reload all products",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics",
    )
    
    args = parser.parse_args()
    
    if not args.sync and not args.full and not args.stats:
        parser.print_help()
        sys.exit(1)
    
    if args.stats:
        show_stats()
        return
    
    # Import here to avoid slow startup for --stats
    from src.application.pipeline import DataIngester
    
    ingester = DataIngester()
    
    try:
        if args.full:
            print("Starting full data ingestion...")
            count = ingester.ingest_all_sync(drop_existing=True)
            print(f"Done! Loaded {count} products into Milvus")
        else:
            # Smart sync: check if DB has data
            current_count = get_product_count()
            
            if current_count == 0:
                print("Database empty. Starting full data ingestion...")
                count = ingester.ingest_all_sync(drop_existing=False)
                print(f"Done! Loaded {count} products into Milvus")
            else:
                print(f"Found {current_count} products. Starting incremental update...")
                result = ingester.incremental_update_sync()
                print(f"Done! Inserted: {result['inserted']}, Updated: {result['updated']}, Unchanged: {result.get('unchanged', 0)}, Deleted: {result['deleted']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
