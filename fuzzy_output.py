class FuzzyOutput:
    def __init__(self, name, xs, ys):
        self.name = name
        self.xs = xs
        self.ys = ys
        self.mu = 0
        cs = [x for x, y in zip(xs, ys) if y == 1]
        self.c = sum(cs) / len(cs)

    def __repr__(self):
        return self.name