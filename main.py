# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import string
import random
from utils import unicodeToAscii
from train import trainIters
from test import evaluate, evaluateRandomly

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import prepareData

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from models import EncoderRNN, AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

def main():

    # TODO CHECK THE EFFECT OF LOADING DATA HERE:
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    lang_pack = input_lang, output_lang, pairs
    print(random.choice(pairs))

    hidden_size = 2
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 100, print_every=5000, plot_every=1, lang_pack=lang_pack)

    ######################################################################
    #

    evaluateRandomly(encoder1, attn_decoder1, lang_pack=lang_pack)

    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "je suis trop froid .", lang_pack=lang_pack)

    ######################################################################
    # For a better viewing experience we will do the extra work of adding axes
    # and labels:
    #

    def showAttention(input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()



    def evaluateAndShowAttention(input_sentence):
        output_words, attentions = evaluate(
            encoder1, attn_decoder1, input_sentence, lang_pack=lang_pack)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)

    evaluateAndShowAttention("elle a cinq ans de moins que moi .")
    evaluateAndShowAttention("elle est trop petit .")
    evaluateAndShowAttention("je ne crains pas de mourir .")
    evaluateAndShowAttention("c est un jeune directeur plein de talent .")


if __name__ == "__main__":
    main()
