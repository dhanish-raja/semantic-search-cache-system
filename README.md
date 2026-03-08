Threshold Behavior

0.95 → strict cache
0.88 → balanced
0.75 → aggressive caching

Docker Setup

Build image:
docker build -t semantic-cache-system .

Run container:
docker run -p 8000:8000 semantic-cache-system