class User:
    # user

    def __init__(self, start_position, start_dt, speed):
        self.position = start_position
        self.dt = start_dt
        self.speed = speed
        self.neighbors = []
        self.destinations = []