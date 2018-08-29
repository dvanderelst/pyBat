class Controller:
    def __init__(self, kp, kd):
        self.previous_error = None
        self.kp = kp
        self.kd = kd

    def get_output(self,error, dt):
        if self.previous_error is None: self.previous_error = error
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.kd * derivative
        self.previous_error = error
        return output

    def reset(self):
        self.previous_error = None
