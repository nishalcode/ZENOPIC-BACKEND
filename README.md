# ZENOPIC-BACKEND

## Description
ZENOPIC-BACKEND is a lightweight backend for generating images.  
It is designed to be fast, secure, and easy to deploy.  

**⚠️ Note:** This project is **proprietary**. No one is allowed to copy, reuse, or redistribute the backend without explicit permission from the owner (Nishal). All rights reserved.

## Features
- REST API endpoints for image generation
- Easy to deploy with Python or Docker
- Structured and scalable code for future upgrades

## Setup

1. Clone the repo (for authorized users only):
```bash
git clone https://github.com/nishalcode/ZENOPIC-BACKEND.git

2. Install dependencies:



pip install -r requirements.txt

3. Copy .env.example to .env and add your API keys/secrets:



cp .env.example .env

Running the Backend

Run locally with Python:

python main.py

Or run with Docker:

docker build -t zenopic-backend .
docker run -p 8000:8000 zenopic-backend

API Endpoints

GET / – Check server status

POST /generate – Generate an image (provide payload)


#License

This project is proprietary.
All rights reserved. Unauthorized copying, modification, or distribution is strictly prohibited.
