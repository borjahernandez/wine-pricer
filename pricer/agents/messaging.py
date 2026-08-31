"""Telling you about a find. Pushover if it is configured, otherwise the log.

Deliberately degradable: a missing notification key should never stop a planning run.
"""

import os

import requests

from pricer.agents.agent import Agent

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    colour = "\033[95m"

    def __init__(self):
        self.token = os.environ.get("PUSHOVER_TOKEN")
        self.user = os.environ.get("PUSHOVER_USER")
        self.enabled = bool(self.token and self.user)
        self.log("Pushover configured" if self.enabled else "No Pushover keys, notifications go to the log only")

    def notify(self, title: str, message: str) -> None:
        self.log(f"{title} -- {message}")
        if not self.enabled:
            return
        try:
            requests.post(
                PUSHOVER_URL,
                data={"token": self.token, "user": self.user, "title": title, "message": message},
                timeout=10,
            )
        except requests.RequestException as error:
            self.log(f"Pushover failed: {error}")
