from UPISAS.strategy import Strategy
import numpy as np
from UPISAS.strategies.DQN import DQN
from UPISAS.strategies.SignalBased import SignalBased
import tqdm

class DingNetStrategy(Strategy):

    def __init__(self, exemplar, mode="learning", algorithm="DQN_discrete"):
        super().__init__(exemplar)
        self.steps_per_episode = 2000
        self.mode = mode
        self.adaptation_step = 0
        self.threshold = -40.0
        self.algorithm = algorithm
        self.planner = None
        self.progress_bar = None
        self.episode = 0
        self.episode_reward = 0

        if self.mode == "learning":
            self.exemplar.reset_mote()
            self.exemplar.start_simulation()

    def analyze(self):
        if 'status' in self.knowledge.monitored_data:
            if self.knowledge.monitored_data['status'] != "RUNNING":
                self.knowledge.analysis_data['operation'] = "restart"
                self.episode_reward = 0
                self.episode = 0
                self.adaptation_step = 0
                if self.progress_bar is not None:
                    self.progress_bar.close()
                return True
        if self.mode == "learning":
            # waiting until mote's state changed
            if len(self.knowledge.monitored_data["last_motes"]) == 0:
                return False
            if self.knowledge.monitored_data["last_motes"][0]["bestSignalStrength"] < -500:
                return False
            last_signal_strength = self.knowledge.monitored_data["last_motes"][0]["bestSignalStrength"]
            current_signal_strength = self.knowledge.monitored_data["motes"][0]["bestSignalStrength"]
            # Mote's state changed
            if last_signal_strength != current_signal_strength:
                # Construct observation
                EUI = self.knowledge.monitored_data["motes"][0]["EUI"]
                obervervation = self._normalized_observation(EUI)
                self.knowledge.analysis_data["observations"] = dict()
                self.knowledge.analysis_data["observations"][EUI] = obervervation
                # Construct reward
                self.knowledge.analysis_data["rewards"] = dict()
                self.knowledge.analysis_data["rewards"][EUI] = -abs(current_signal_strength - self.threshold)
                # print("[Analysis]\tobservation: " + str(obervervation))
                # print("[Analysis]\treward: " + str(-abs(current_signal_strength - self.threshold)))
                return True

        elif self.mode == "normal":
            if self.algorithm == "DQN_discrete":
                motes = self.knowledge.monitored_data["motes"]
                self.knowledge.analysis_data["observations"] = dict()
                self.knowledge.analysis_data["rewards"] = dict()
                self.knowledge.analysis_data["actions"] = dict()
                for mote in motes:
                    EUI = mote["EUI"]
                    obervervation = self._normalized_observation(EUI)
                    self.knowledge.analysis_data["observations"][EUI] = obervervation
                    self.knowledge.analysis_data["rewards"][EUI] = -abs(mote["bestSignalStrength"] - self.threshold)
                    self.knowledge.analysis_data["actions"][EUI] = mote["transmissionPower"]

                # print(f"\r[Analysis]: Rewards: " + str(self.knowledge.analysis_data["rewards"]), end="")
                return True
            elif self.algorithm == "Signal_based":
                motes = self.knowledge.monitored_data["motes"]
                self.knowledge.analysis_data["signal_strength"] = dict()
                self.knowledge.analysis_data["rewards"] = dict()
                self.knowledge.analysis_data["transmission_power"] = dict()
                for mote in motes:
                    EUI = mote["EUI"]
                    self.knowledge.analysis_data["signal_strength"][EUI] = mote["bestSignalStrength"]
                    self.knowledge.analysis_data["rewards"][EUI] = -abs(mote["bestSignalStrength"] - self.threshold)
                    self.knowledge.analysis_data["transmission_power"][EUI] = mote["transmissionPower"]

                # print(f"\r[Analysis]: Rewards: " + str(self.knowledge.analysis_data["rewards"]), end="")
                return True
        return False

    def plan(self):
        if "operation" in self.knowledge.analysis_data and self.knowledge.analysis_data["operation"] == "restart":
            self.knowledge.analysis_data["operation"] = None
            self.knowledge.plan_data["operation"] = "restart"
            return True

        if self.planner is None:
            if self.algorithm == "DQN_discrete":
                self.planner = DQN(n_actions=101, n_observations=10)
            elif self.algorithm == "Signal_based":
                self.planner = SignalBased(self.threshold)
        
        if self.mode == "learning":
            # One episode is finished
            if self.progress_bar is None:
                self.progress_bar = tqdm.tqdm(total=self.steps_per_episode)
            if self.adaptation_step == self.steps_per_episode:
                self.progress_bar.close()
                self.progress_bar = tqdm.tqdm(total=self.steps_per_episode)
                self.episode += 1
                self.adaptation_step = 0
                self.episode_reward = 0
                self.knowledge.plan_data["operation"] = "restart"
                self.planner.state = None
            
            # Start a step
            state = self.knowledge.analysis_data["observations"][list(self.knowledge.analysis_data["observations"].keys())[0]]
            try:
                state = np.array(state, dtype=np.float32)
            except:
                print(self.knowledge.analysis_data["observations"])

            reward = self.knowledge.analysis_data["rewards"][list(self.knowledge.analysis_data["rewards"].keys())[0]]
            new_action = self.planner.learn(state, reward)
            if new_action == None:
                return False
            # print("[Plan]\taction: " + str(new_action))
            self.knowledge.plan_data["moteOptions"] = list()
            self.knowledge.plan_data["moteOptions"].append(dict())
            self.knowledge.plan_data["moteOptions"][0]["EUI"] = list(self.knowledge.analysis_data["observations"].keys())[0] 
            self.knowledge.plan_data["moteOptions"][0]["transmissionPower"] = new_action
            self.adaptation_step += 1
            self.episode_reward += reward
            self.progress_bar.update(1)
            self.progress_bar.set_description("Episode: " + str(self.episode) + " | reward: " + str(self.episode_reward / self.adaptation_step))

        elif self.mode == "normal":
            if self.algorithm == "DQN_discrete":
                observations = self.knowledge.analysis_data["observations"]
                self.knowledge.plan_data["moteOptions"] = list()
                for EUI in observations:
                    state = observations[EUI]
                    new_action = self.planner.predict(state)
                    self.knowledge.plan_data["moteOptions"].append(dict())
                    self.knowledge.plan_data["moteOptions"][-1]["EUI"] = EUI
                    self.knowledge.plan_data["moteOptions"][-1]["transmissionPower"] = new_action
                    self.adaptation_step += 1
                    self.episode_reward += self.knowledge.analysis_data["rewards"][EUI]
            elif self.algorithm == "Signal_based":
                signal_strength = self.knowledge.analysis_data["signal_strength"]
                transmission_power = self.knowledge.analysis_data["transmission_power"]
                self.knowledge.plan_data["moteOptions"] = list()
                for EUI in signal_strength:
                    state = signal_strength[EUI]
                    new_action = self.planner.predict(state, transmission_power[EUI])
                    self.knowledge.plan_data["moteOptions"].append(dict())
                    self.knowledge.plan_data["moteOptions"][-1]["EUI"] = EUI
                    self.knowledge.plan_data["moteOptions"][-1]["transmissionPower"] = new_action
                    self.adaptation_step += 1
                    self.episode_reward += self.knowledge.analysis_data["rewards"][EUI]
            print(f"\r[Plan]: reward: {self.episode_reward / self.adaptation_step}", end="")
        return True

    def _normalized_observation(self, EUI):
        index = 0
        for mote in self.knowledge.monitored_data["motes"]:
            if mote["EUI"] == EUI:
                index = self.knowledge.monitored_data["motes"].index(mote)
        # Construct observation
        latitude = self.knowledge.monitored_data["motes"][index]["latitude"]
        longitude = self.knowledge.monitored_data["motes"][index]["longitude"]
        transmission_power = self.knowledge.monitored_data["motes"][index]["transmissionPower"]
        movement_speed = self.knowledge.monitored_data["motes"][index]["movementSpeed"]
        signal_strength = self.knowledge.monitored_data["motes"][index]["bestSignalStrength"]
        gateway_distance = self.knowledge.monitored_data["motes"][index]["bestGatewayDistance"]
        gateway_latitude = self.knowledge.monitored_data["motes"][index]["bestGatewayLatitude"]
        gateway_longitude = self.knowledge.monitored_data["motes"][index]["bestGatewayLongitude"]
        path_loss = self.knowledge.monitored_data["motes"][index]["pathLoss"]
        shadow_fading = self.knowledge.monitored_data["motes"][index]["shadowFading"]

        # Normalize observation
        min_latitude = self.knowledge.monitored_data["minMapPosition"]["latitude"]
        min_longitude = self.knowledge.monitored_data["minMapPosition"]["longitude"]
        max_latitude = self.knowledge.monitored_data["maxMapPosition"]["latitude"]
        max_longitude = self.knowledge.monitored_data["maxMapPosition"]["longitude"]
        max_distance = self.knowledge.monitored_data["maxDistance"]
        latitude = (latitude - min_latitude) / (max_latitude - min_latitude)
        longitude = (longitude - min_longitude) / (max_longitude - min_longitude)
        transmission_power = (transmission_power - (-50)) / (50 - (-50))
        movement_speed = (movement_speed - 1) / (51 - 1)
        signal_strength = (signal_strength - (-130)) / (30 - (-130))
        gateway_distance = (gateway_distance - 0) / (max_distance - 0)
        gateway_latitude = (gateway_latitude - min_latitude) / (max_latitude - min_latitude)
        gateway_longitude = (gateway_longitude - min_longitude) / (max_longitude - min_longitude)
        path_loss = (path_loss - 0) / (3 - 0)
        shadow_fading = (shadow_fading - 0) / (3 - 0)

        return [latitude, longitude, transmission_power, movement_speed, signal_strength, gateway_distance,
                gateway_latitude, gateway_longitude, path_loss, shadow_fading]