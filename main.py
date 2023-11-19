import streamlit as st
from GanColorizer import ColorizerModel
import torch

model = ColorizerModel()

model.load_weights()

image = model.generate_color('download.png')
