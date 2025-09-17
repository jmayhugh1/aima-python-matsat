class WumpusPosition:
    def __init__(self, x, y, orientation):
        self.X = x
        self.Y = y
        self.orientation = orientation

    def get_location(self):
        return self.X, self.Y

    def set_location(self, x, y):
        self.X = x
        self.Y = y

    def get_orientation(self):
        return self.orientation

    def set_orientation(self, orientation):
        self.orientation = orientation

    def __eq__(self, other):
        if (
            other.get_location() == self.get_location()
            and other.get_orientation() == self.get_orientation()
        ):
            return True
        else:
            return False

    def __lt__(self, other):
        if self.X != other.X:
            return self.X < other.X
        if self.Y != other.Y:
            return self.Y < other.Y
        return self.orientation < other.orientation

    def __hash__(self):
        return hash((self.X, self.Y, self.orientation))

    def __str__(self):
        return f"([{self.X}, {self.Y}], {self.orientation})"
