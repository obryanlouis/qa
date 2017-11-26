"""Provides a way to tokenize text with Stanford CoreNLP.
"""

import json
import os
import preprocessing.constants as constants
import re
import subprocess

from preprocessing.tokenized_word import *
from pycorenlp import StanfordCoreNLP

class StanfordCoreNlpCommunication():
    def __init__(self, download_dir):
        self.server_process = None
        self.download_dir = download_dir
        self.nlp = None

    def start_server(self):
        command = [ "java", "-cp",
        os.path.join(self.download_dir,
                "stanford-corenlp-full-2017-06-09/*"),
        "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-port", constants.STANFORD_CORENLP_PORT,
        "-quiet"]
        self.server_process = subprocess.Popen(command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        if self.server_process.poll() is not None:
            raise Exception("Couldn't start Stanford CoreNLP server.")
        print("Started Stanford CoreNLP server on port " + constants.STANFORD_CORENLP_PORT)
        self.nlp = StanfordCoreNLP('http://localhost:' + constants.STANFORD_CORENLP_PORT)

    def stop_server(self):
        print("Killed Stanford CoreNLP server")
        self.server_process.kill()

    def tokenize_list(self, text_list):
        output_list = []
        i = 0
        for text in text_list:
            output_list.append(self.tokenize_text(text))
            i += 1
            print("Progress", i, "out of", len(text_list))
        return output_list

    def _get_tokenized_words(self, annotation):
        tokens = []
        for sentence in annotation["sentences"]:
            for token in sentence["tokens"]:
                tokens.append(TokenizedWord(
                    token["word"],
                    token["characterOffsetBegin"],
                    token["characterOffsetEnd"],
                    token["ner"],
                    token["pos"]))
        return tokens

    def tokenize_text(self, text):
        """Input: A string

           Output: A list of TokenizedWord's from the text.
        """
        annotate = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,pos,ner',
            'outputFormat': 'json',
            'tokenize.language': 'English',
            'splitHyphenated': True,
            'tokenize.options': 'splitHyphenated=true,untokenizable=allKeep,invertible=true',
        })
        if isinstance(annotate, str):
            print("Some internal failure happened when using Stanford CoreNLP")
            return None
        return self._get_tokenized_words(annotate)
