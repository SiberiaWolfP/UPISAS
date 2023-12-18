from abc import ABC, abstractmethod
import requests
import pprint

from UPISAS.exceptions import EndpointNotReachable, ServerNotReachable
from UPISAS.knowledge import Knowledge
from UPISAS import validate_schema, get_response_for_get_request
import logging

pp = pprint.PrettyPrinter(indent=4)
s = requests.Session()

class Strategy(ABC):

    def __init__(self, exemplar):
        self.exemplar = exemplar
        self.knowledge = Knowledge(dict(), dict(), dict(), dict(), dict(), dict(), dict())

    def ping(self):
        ping_res = self._perform_get_request(self.exemplar.base_endpoint)
        logging.info(f"ping result: {ping_res}")

    def monitor(self, endpoint_suffix="monitor", with_validation=False):
        fresh_data = self._perform_get_request(endpoint_suffix)
        # print("[Monitor]\tgot fresh_data: " + str(fresh_data))
        if with_validation:
            validate_schema(fresh_data, self.knowledge.monitor_schema)
        data = self.knowledge.monitored_data
        for key in list(fresh_data.keys()):
            if key not in data:
                data[key] = []
                data['last_' + key] = None
            data['last_' + key] = data[key]
            data[key] = fresh_data[key]
        # print("[Knowledge]\tdata monitored: " + str(self.knowledge.monitored_data))
        return True

    def execute(self, adaptation, endpoint_suffix="execute", with_validation=False):
        if with_validation:
            validate_schema(adaptation, self.knowledge.execute_schema)
        url = '/'.join([self.exemplar.base_endpoint, endpoint_suffix])
        response = s.put(url, json=adaptation)
        # print("[Execute]\tposted configuration: " + str(adaptation))
        if response.status_code != 200:
            logging.error("Cannot execute adaptation on remote system. " + response.text)
            raise EndpointNotReachable
        if "operation" in self.knowledge.plan_data and self.knowledge.plan_data["operation"] == "restart":
            if self.mode == "learning":
                self.exemplar.reset_mote()
            self.exemplar.start_simulation()
            self.knowledge.plan_data["operation"] = None
        return True

    def get_adaptation_options(self, endpoint_suffix: str = "adaptation_options", with_validation=False):
        self.knowledge.adaptation_options = self._perform_get_request(endpoint_suffix)
        if with_validation:
            validate_schema(self.knowledge.adaptation_options, self.knowledge.adaptation_options_schema)
        logging.info("adaptation_options set to: ")
        pp.pprint(self.knowledge.adaptation_options)

    def get_monitor_schema(self, endpoint_suffix = "monitor_schema"):
        self.knowledge.monitor_schema = self._perform_get_request(endpoint_suffix)
        logging.info("monitor_schema set to: ")
        pp.pprint(self.knowledge.monitor_schema)

    def get_execute_schema(self, endpoint_suffix = "execute_schema"):
        self.knowledge.execute_schema = self._perform_get_request(endpoint_suffix)
        logging.info("execute_schema set to: ")
        pp.pprint(self.knowledge.execute_schema)

    def get_adaptation_options_schema(self, endpoint_suffix: str = "adaptation_options_schema"):
        self.knowledge.adaptation_options_schema = self._perform_get_request(endpoint_suffix)
        logging.info("adaptation_options_schema set to: ")
        pp.pprint(self.knowledge.adaptation_options_schema)

    def _perform_get_request(self, endpoint_suffix: str):
        url = '/'.join([self.exemplar.base_endpoint, endpoint_suffix])
        response = get_response_for_get_request(url)
        if response.status_code == 404:
            logging.error("Please check that the endpoint you are trying to reach actually exists.")
            raise EndpointNotReachable
        return response.json()

    @abstractmethod
    def analyze(self):
        """ ... """
        pass

    @abstractmethod
    def plan(self):
        """ ... """
        pass

