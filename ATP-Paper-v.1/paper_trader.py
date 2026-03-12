class PaperTrader:

    def __init__(self, balance):
        self.balance = balance
        self.position = 0

    def execute(self, signal, price):

        if signal == "BUY" and self.balance > 0:

            self.position = self.balance / price
            self.balance = 0

        elif signal == "SELL" and self.position > 0:

            self.balance = self.position * price
            self.position = 0