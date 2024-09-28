class FuzzyInput:
    def __init__(self, name, xs, ys, x0):
        self.name = name
        self.xs = xs
        self.ys = ys
        self.setX(x0)
        
    def calcMu(self, x0):
        if x0 < self.xs[0]:
            return self.ys[0]
        elif x0 > self.xs[-1]:
            return self.ys[-1]
        
        for i, (x1, x2) in enumerate(zip(self.xs[:-1], self.xs[1:])):
            if (x0 >= x1 and x0 < x2):
                return FuzzyInput.line_through_two_points(x1, self.ys[i], x2, self.ys[i+1])(x0)
    
    def setX(self, x0):
        self.mu = self.calcMu(x0)

    def __repr__(self):
        return self.name

    @staticmethod
    def line_through_two_points(x1, y1, x2, y2):
        return lambda x: y1 + (y2 - y1)/(x2 - x1) * (x - x1)