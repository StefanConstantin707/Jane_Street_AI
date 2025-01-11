from ModelTest.NNTest import nn_test, nn_test_gpu, nn_test_online, nn_test_lag_responder, nn_test_lag_responder_mapping
from ModelTest.SimpleTransformerTest import attention_test_symbols, attention_test
from ModelTest.rnnTest import rnn_test


def main():
    nn_test_lag_responder_mapping()

if __name__ == '__main__':
    main()
