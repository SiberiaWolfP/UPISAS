# from UPISAS.example_strategy import ExampleStrategy
from UPISAS.exemplar import Exemplar
# from UPISAS.exemplars.swim import SWIM
from UPISAS.exemplars.Dingnet import Dingnet_Exemplar
from UPISAS.strategies.dingnet_strategy import DingNetStrategy


import signal
import sys


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    exemplar.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
if __name__ == '__main__':
    
    # exemplar = SWIM(auto_start=True)
    # exemplar.start_run()

    # ding_args = ["basic_graph.xml", "Signal-based", "ReliableEfficient", "1"]
    exemplar = Dingnet_Exemplar(auto_start=True, debug=True)
    exemplar.start_run([])
    exemplar.init_world(config_name="test_2_5gateways_1mote.xml", speed=5)
    exemplar.start_simulation()
    try:
        strategy = DingNetStrategy(exemplar, mode="normal", algorithm="DQN_discrete")
        # strategy = DingNetStrategy(exemplar, mode="learning", algorithm="DQN_discrete")
        # strategy = DingNetStrategy(exemplar, mode="normal", algorithm="Signal_based")

        while True:
            strategy.monitor()
            if strategy.analyze():
                if strategy.plan():
                    strategy.execute(strategy.knowledge.plan_data)
    except:
        exemplar.stop()
        sys.exit(0)