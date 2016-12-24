
class Reader:
    def __init__(self):
        self.f = open('cuo.txt', 'r+')
        self.lines = []

    def read(self):
        for line in self.f:
            self.lines.append(line)
        self.f.close()
        return self.lines
