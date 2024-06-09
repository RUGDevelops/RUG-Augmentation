import threading
from PIL import Image
from pathlib import Path
from flask import Flask, request, Response, jsonify
from ModelCreator import ModelCreator

def test_addition():
    assert 5 + 5 == 10