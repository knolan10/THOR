import time
import requests
from astropy.time import Time


class RateLimiter:
    """Allow at most max_calls per period seconds."""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._calls = 0
        self._window_start = None

    def __call__(self):
        if self._window_start is None:
            self._window_start = time.monotonic()
        self._calls += 1
        if self._calls >= self.max_calls:
            elapsed = time.monotonic() - self._window_start
            if elapsed < self.period:
                time.sleep(self.period - elapsed)
            self._calls = 0
            self._window_start = time.monotonic()

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

    def generic_slack_message(self, message: str):
        """
        send a generic slack message
        """
        self.send_slack(message)

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


# fritz API
def post_candidate_to_fritz(object_id, ra, dec, time, filter_ids, broker_id, group_ids, token):
    # https://docs.fritz.science/api.html#tag/candidates/POST/api/candidates/
    base = "https://fritz.science/api"
    headers = {"Content-Type": "application/json", "Authorization": f"token {token}"}

    payload = {
        "id": object_id,
        "filter_ids": list(filter_ids),
        "passed_at": time,
    }
    r = requests.post(f"{base}/candidates", headers=headers, json=payload)
    if r.status_code == 400 and "must not be null for a new Obj" in r.text:
        # ra/dec only required for object Fritz doesnt know yet
        # in this case the healpix is computed from those same coords so it stays consistent
        r = requests.post(f"{base}/candidates", headers=headers, 
                          json={**payload, "ra": ra, "dec": dec}) 
    r.raise_for_status()

    # this endpoint normally saves as a source, but it only creates the Source row when it also has to create the Obj.
    # so here it just pulls in the photometry and the three cutouts and leaves it unsaved, ready to scan. 
    r = requests.post(
        f"{base}/brokers/{broker_id}/alerts/{object_id}/save",
        headers=headers,
        json={"group_ids": list(group_ids)},
    )
    print(r.text)
    if r.status_code == 400 and "is not allowed" in r.text:
        print(f"Warning: broker save skipped for {object_id} — bad filter in Fritz alert data")
    else:                                                                                                                                                                                
        r.raise_for_status()


def post_auto_annotation_fritz(object_id, catalog_id, catalog_dict, group_ids, fritz_token):
    # https://docs.fritz.science/api.html#tag/annotations/POST/api/{associated_resource_type}/{resource_id}/annotations
    catalog_data = catalog_dict[catalog_id]

    url = f"https://fritz.science/api/sources/{object_id}/annotations"

    payload = {
        "origin": catalog_id,
        "data": catalog_data,
        "group_ids": group_ids
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {fritz_token}"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()