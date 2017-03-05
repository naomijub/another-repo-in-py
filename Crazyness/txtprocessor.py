

class TextProcessor:
    def __init__(self):
        self.rows = []

    def process(self, lines):
        for line in lines:
            aux = line[1:line.find(']')].split(' ')
            for var in line[line.find(']') + 2:line.find('\r')].split('\t'):
                aux.append(var)
            self.rows.append(aux)

        return self.rows
