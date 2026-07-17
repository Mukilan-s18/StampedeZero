"""
StampedeZero - SMS Gateway Verification Script
================================================
Run this script FIRST to verify that the Twilio SMS pipeline is operational.
A successful test message confirms the alert gateway is live and ready
for integration with the predictive engine.

Usage:
    python sms_tester.py

Prerequisites:
    1. Copy .env.example to .env
    2. Fill in your Twilio credentials (SID, Token, From number)
    3. Add your target phone number
    4. Ensure the target number is verified in Twilio (trial accounts only)
"""

import os
import sys
from twilio.rest import Client
from dotenv import load_dotenv


def verify_env_vars():
    """Validate that all required environment variables are present."""
    required_vars = ["TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM", "TARGET_PHONE"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("   Please copy .env.example to .env and fill in your credentials.")
        sys.exit(1)


def send_test_sms():
    """Send a single test SMS to verify the Twilio gateway is operational."""
    load_dotenv()
    verify_env_vars()

    client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))

    print("📡 Sending test message via Twilio...")
    try:
        message = client.messages.create(
            body="🚨 StampedeZero Test: Communication line open. Alert gateway verified.",
            from_=os.getenv("TWILIO_FROM"),
            to=os.getenv("TARGET_PHONE"),
        )
        print(f"✅ Message sent successfully!")
        print(f"   SID:    {message.sid}")
        print(f"   Status: {message.status}")
        print(f"   To:     {os.getenv('TARGET_PHONE')}")
    except Exception as e:
        print(f"❌ SMS delivery failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    send_test_sms()
