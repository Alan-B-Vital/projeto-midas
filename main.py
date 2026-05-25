from tests.static_test import static_test
from config import MODEL_NAME
from setup import init

if __name__ == '__main__':
    # Rodar para baixar os indicadores treinar o modelo, uma vez treinado pode deixar comentado, não será iniciado novamente
    # init()
    

    # Roda um teste com dados dos indicadores, de preferencia um que foi treinado:
    # ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOG', 'JPM', 'XOM', 'WMT']
    static_test(MODEL_NAME, 'AAPL')
