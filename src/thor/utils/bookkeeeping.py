import requests
from astropy.time import Time

class Slacker:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url

    def send_slack(self, message: str):
        """
        Send a slack message
        :param message: message to send
        :return: None
        """
        message_payload = {"text": message}
        try:
            response = requests.post(self.webhook_url, json=message_payload)
            if response.status_code == 200:
                print("Message was sent successfully to slack")
            else:
                print(
                    "Message was not sent to slack, status_code:", response.status_code
                )
        except Exception as e:
            print("An exception occurred while sending a slack message", e)

    def slack_crossmatch_results(
        self, initial_alert_count, candidate_count
    ):
        """
        format message summarizing LSST crossmatch results
        """
        day = Time.now().strftime("%Y-%m-%d")
        jd = round(Time.now().jd)
        message = f"*Crossmatch_alerts successfully ran on {day} (JD={jd}). From {initial_alert_count} astrophysical alerts, {candidate_count} candidates sent to Fritz for scanning*"
        self.send_slack(message)