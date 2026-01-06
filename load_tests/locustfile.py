"""
Load testing for Product Research Assistant API.

Run with Web UI (Default):
    uv run locust -f load_tests/locustfile.py --host=http://localhost:8000
    # Then open http://localhost:8089 to set user count/spawn rate

Run Headless (Command Line) + Save CSV:
    uv run locust -f load_tests/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 --run-time 1m --csv=load_tests/my_results

Args:
    -u, --users: Peak number of concurrent users
    -r, --spawn-rate: Rate to spawn users users/second
    --run-time: Stop after the specified time (e.g. 10s, 20m)
    --csv: Prefix for the CSV files (e.g. 'my_results' -> 'my_results_stats.csv')

This simulates users making queries to the API.
Task weights determine how often each action is performed:
- Weight 3: product_catalog_rag (most common)
- Weight 2: price_analysis
- Weight 1: web_search, multi-tool, health, history
"""

from locust import HttpUser, task, between


class ProductResearchUser(HttpUser):
    """Simulated user making API requests."""
    
    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)
    
    @task(3)  # Most common - 3x weight
    def query_product_rag(self):
        """Search product catalog."""
        self.client.post(
            "/query",
            json={"query": "What wireless headphones do we have in stock?"},
        )
    
    @task(2)  # Common - 2x weight
    def query_price_analysis(self):
        """Analyze pricing."""
        self.client.post(
            "/query",
            json={"query": "Which products have lowest profit margins?"},
        )
    
    @task(1)  # Less common - uses external API
    def query_web_search(self):
        """Search the web."""
        self.client.post(
            "/query",
            json={"query": "Current market price for noise-cancelling headphones?"},
        )
    
    @task(1)  # Less common - complex query
    def query_multi_tool(self):
        """Query using multiple tools."""
        self.client.post(
            "/query",
            json={"query": "Should we adjust AudioMax headphones pricing vs competitors?"},
        )
    
    @task(1)
    def health_check(self):
        """Check health endpoint."""
        self.client.get("/health")
    
    @task(1)
    def query_history(self):
        """Check query history."""
        self.client.get("/queries")

# example: จำลอง 100 คน, เพิ่มทีละ 10 คน/วิ, รันนาน 1 นาที
# uv run locust -f load_tests/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 --run-time 1m