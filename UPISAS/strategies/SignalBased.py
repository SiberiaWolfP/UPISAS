class SignalBased():

    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, signal_strength, transmission_power):
        if signal_strength > self.threshold:
            new_transmission_power = transmission_power - 1
            if new_transmission_power < -50:
                new_transmission_power = -50
            return new_transmission_power
        elif signal_strength < self.threshold:
            new_transmission_power = transmission_power + 1
            if new_transmission_power > 50:
                new_transmission_power = 50
            return new_transmission_power
        else:
            return transmission_power