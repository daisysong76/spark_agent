# ai_agent/tools/phone_tool.py
import logging
from twilio.rest import Client
from ai_agent.config.settings import settings

logger = logging.getLogger(__name__)

def make_phone_call(to_phone_number: str, message: str) -> str:
    """
    Initiates a phone call using Twilio, playing a message or TWiML instructions.
    """
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    try:
        call = client.calls.create(
            to=to_phone_number,
            from_=settings.TWILIO_PHONE_NUMBER,
            url=f"http://twimlets.com/message?Message%5B0%5D={message}"
        )
        logger.info(f"Call initiated: {call.sid}")
        return f"Phone call initiated successfully to {to_phone_number}. Call SID: {call.sid}"
    except Exception as e:
        logger.error(f"Error making phone call: {e}", exc_info=True)
        raise RuntimeError(f"Failed to make phone call: {str(e)}")


# We use a logger to record successes and errors.
# We assume the environment variables (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, etc.) are set.
# The message is passed via a Twimlet link for a simple speech message. For advanced use, you can host your own TWiML endpoint.