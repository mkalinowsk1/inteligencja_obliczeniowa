import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forwardPass(wiek, waga, wzrost):
    w_wiek_h1 = -0.46122
    w_waga_h1 = 0.97314
    w_wzrost_h1 = -0.39203
    bias_h1 = 0.80109

    w_wiek_h2 = 0.78548
    w_waga_h2 = 2.10584
    w_wzrost_h2 = -0.57847
    bias_h2 = 0.43529

    w_h1_out = -0.81546
    w_h2_out = 1.03775
    bias_out = -0.2368

    hidden1 = wiek * w_wiek_h1 + waga * w_waga_h1 + wzrost * w_wzrost_h1 + bias_h1
    hidden1_po_aktywacji = sigmoid(hidden1)

    hidden2 = wiek * w_wiek_h2 + waga * w_waga_h2 + wzrost * w_wzrost_h2 + bias_h2
    hidden2_po_aktywacji = sigmoid(hidden2)

    output = hidden1_po_aktywacji * w_h1_out + hidden2_po_aktywacji * w_h2_out + bias_out
    
    return output

print(forwardPass(23, 75, 176))