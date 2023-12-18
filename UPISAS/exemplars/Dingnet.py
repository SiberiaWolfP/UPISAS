import pprint, time

import requests
from UPISAS.exceptions import UPISASException
from UPISAS.exemplar import Exemplar
import logging
pp = pprint.PrettyPrinter(indent=4)
logging.getLogger().setLevel(logging.INFO)

class Dingnet_Exemplar(Exemplar):
    """
    A class which encapsulates a self-adaptive exemplar run in a docker container.
    """
    def __init__(self, auto_start=False, container_name="MyDingnet", debug=False):
        if debug:
            self.debug = True
            self.base_endpoint = "http://localhost:8080"
        else:
            self.debug = False
            my_docker_kwargs = {
                "name":  container_name,    # TODO add your container name
                "image": "dingnet:latest", # TODO add your exemplar's image name
                "ports" : {8080: 8080}}              # TODO add any other necessary ports

            super().__init__("http://localhost:8080", my_docker_kwargs, auto_start)
    
    def start_run(self, args):
        if self.debug:
            return
        # args = ["basic_graph.xml", "Signal-based", "ReliableEfficient", "5"]
        args_str = ' '.join(args)
        cmd = f'sh -c "java -jar /app.jar {args_str}"'
        self.exemplar_container.exec_run(cmd= cmd, detach=True)

    def init_world(self, config_name="mock.xml", speed=5):
        self.config_name = config_name
        self.speed = speed
        request_url = self.base_endpoint + "/init_world"
        # print("[DingNet]\t" + "Init world\t" + "config name: " + config_name + ", speed: " + str(speed))
        response = requests.post(request_url, json={"configName": config_name, "speed": speed})
        if response.status_code != 200:
            logging.error("Cannot init world")
            raise UPISASException
        return True
        

    def start_simulation(self):
        request_url = self.base_endpoint + "/start"
        # print("[DingNet]\t" + "Start simulation")
        response = requests.get(request_url)
        if response.status_code != 200:
            logging.error("Cannot start simulation")
            raise UPISASException
        time.sleep(1)
        return True
    
    def reset_map(self):
        request_url = self.base_endpoint + "/reset_map"
        # print("[DingNet]\t" + "Reset map")
        response = requests.get(request_url)
        if response.status_code != 200:
            logging.error("Cannot reset map")
            raise UPISASException
        return True
    
    def reset_entities(self):
        request_url = self.base_endpoint + "/reset_entities"
        # print("[DingNet]\t" + "Reset entities")
        response = requests.get(request_url)
        if response.status_code != 200:
            logging.error("Cannot reset entities")
            raise UPISASException
        return True

    def reset_gateways(self):
        request_url = self.base_endpoint + "/reset_gateways"
        # print("[DingNet]\t" + "Reset gateways")
        response = requests.get(request_url)
        if response.status_code != 200:
            logging.error("Cannot reset gateways")
            raise UPISASException
        return True
    
    def reset_mote(self):
        request_url = self.base_endpoint + "/reset_mote"
        # print("[DingNet]\t" + "Reset mote")
        response = requests.get(request_url)
        if response.status_code != 200:
            logging.error("Cannot reset mote")
            raise UPISASException
        return True