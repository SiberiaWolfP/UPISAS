from UPISAS.exemplars.Dingnet import Dingnet_Exemplar
from UPISAS.strategies.dingnet_strategy import DingNetStrategy


import signal
import sys
import argparse

def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [arg...]",
        description="Run the DingNet exemplar in a docker container and control it with a self-adaptive strategy",
    )
    parser.add_argument("--debug", help="Run in debug mode", action="store_false")
    parser.add_argument("--config", help="Configuration file name", default="world_1_6g1m.xml")
    parser.add_argument("--speed", help="Simulation speed", default=5)
    parser.add_argument("--algorithm", help="Algorithm to use, select from [DQN_discrete, Signal_based]", default="DQN_discrete")
    parser.add_argument("--mode", help="Mode to use, select from [learning, normal]", default="learning")
    return parser

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    exemplar = Dingnet_Exemplar(auto_start=True, debug=args.debug)
    exemplar.start_run([])
    exemplar.init_world(config_name=args.config, speed=args.speed)
    exemplar.start_simulation()
    try:
        strategy = DingNetStrategy(exemplar, mode=args.mode, algorithm=args.algorithm)

        while True:
            strategy.monitor()
            if strategy.analyze():
                if strategy.plan():
                    strategy.execute(strategy.knowledge.plan_data)
    except:
        sys.exit(0)