
## Description
ZENOPIC-BACKEND is a lightweight backend for image generation and other API tasks.  
It is designed to be fast, secure, and easy to deploy.

## Features
- REST API endpoints for image generation
- Easy to deploy with Python or Docker
- Structured and scalable for future upgrades

##License / Proprietary Notice

This project is proprietary.
All rights are reserved by the owner (Nishal).
No one is allowed to copy, reuse, redistribute, or modify this backend without explicit permission

## Setup

### 1. Clone the Repo
For authorized users only:
```bash
git clone https://github.com/nishalcode/ZENOPIC-BACKEND.git

2. Install Dependencies

pip install -r requirements.txt

3. Environment Variables

Copy the example environment file and add your keys:

cp .env.example .env

Running the Backend

Run locally with Python:

python main.py

Or run with Docker:

docker build -t zenopic-backend .
docker run -p 8000:8000 zenopic-backend

API Endpoints

GET / – Check server status

POST /generate – Generate an image (send JSON payload).
