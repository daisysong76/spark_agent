# ai_agent/tools/email_tool.py
import logging
import requests
from ai_agent.config.settings import settings

logger = logging.getLogger(__name__)

def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Sends an email using SendGrid's API.
    """
    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {settings.SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "personalizations": [
            {"to": [{"email": recipient}]}
        ],
        "from": {"email": settings.EMAIL_FROM},
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": body}
        ]
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        resp.raise_for_status()
        logger.info(f"Email sent to {recipient}, resp={resp.status_code}")
        return f"Email sent to {recipient} successfully."
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send email: {e}", exc_info=True)
        raise RuntimeError(f"Failed to send email: {str(e)}")


# We do a POST to SendGrid’s /mail/send with the API key from environment variables.
# We handle potential RequestException.
# In production, you might want to implement HTML email or templating.