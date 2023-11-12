import pprint, time
from UPISAS.exemplar import Exemplar
import logging
pp = pprint.PrettyPrinter(indent=4)
logging.getLogger().setLevel(logging.INFO)

class YourExemplar(Exemplar):
    """
    A class which encapsulates a self-adaptive exemplar run in a docker container.
    """
    def __init__(self, auto_start=False, container_name="Dingnet"):
        my_docker_kwargs = {
            "name":  container_name,    # TODO add your container name
            "image": "xpsky/dingnet", # TODO add your exemplar's image name
            "ports" : {8080: 8080}}              # TODO add any other necessary ports

        super().__init__("http://localhost:8080", my_docker_kwargs, auto_start)
    
    def start_run(self):
        args = ["basic_graph.xml", "No Adaptation", "ReliableEfficient", "5"]
        cmd = f'sh -c "java -jar /app.jar {args}"'
        self.exemplar_container.exec_run(cmd=cmd, detach=True)
