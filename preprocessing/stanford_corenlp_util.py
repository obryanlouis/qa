"""Provides a way to tokenize text with Stanford CoreNLP.
"""

import json
import os
import preprocessing.constants as constants
import re
import subprocess

from pycorenlp import StanfordCoreNLP

class StanfordCoreNlpCommunication():
    def __init__(self, data_dir):
        self.server_process = None
        self.data_dir = data_dir
        self.nlp = None

    def start_server(self):
        command = [ "java", "-cp",
        os.path.join(self.data_dir,
                "stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar"),
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

    def tokenize_text(self, text):
        """Input: A string

           Output: A json object with the NLP response, including the tokenized
           text and other context.

           Example input: "Sentence 1."
           output:
            [
                {
                    "characterOffsetBegin": 0,
                    "characterOffsetEnd": 8,
                    "index": -1,
                    "originalText": "Sentence",
                    "word": "Sentence"
                },
                {
                    "characterOffsetBegin": 9,
                    "characterOffsetEnd": 10,
                    "index": -1,
                    "originalText": "1",
                    "word": "1"
                },
                {
                    "characterOffsetBegin": 10,
                    "characterOffsetEnd": 11,
                    "index": -1,
                    "originalText": ".",
                    "word": "."
                }
            ]
        """
        annotate = self.nlp.annotate(text, properties={
            'annotators': 'tokenize',
            'outputFormat': 'json',
            'tokenize.language': 'English',
            'splitHyphenated': True,
            'tokenize.options': 'splitHyphenated=true,untokenizable=allKeep,invertible=true',
        })
        if isinstance(annotate, str):
            print("Some internal failure happened when using Stanford CorenNLP")
            return None
        return annotate["tokens"]
