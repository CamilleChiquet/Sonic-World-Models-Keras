# Sonic World-Models with Keras

AI composed of 3 neural networks : a Variational Auto-Encoder, a LSTM and a controller.
An evolutionary algorithm (NEAT) is used to "train" the controller.
For more information about world models architecture, go take a look [here](https://worldmodels.github.io/).

Follow the procedure described in the [procedure.py](https://github.com/CamilleChiquet/Sonic-World-Models-Keras/blob/master/procedure.py) file.

## Before beginning

Copy the "scenario.json" file into the game folder (into the retro package).
I use PyCharm and I've put it into : venv\Lib\site-packages\retro\data\SonicTheHedgehog-Genesis
I only used this AI on the first Sonic game ("SonicTheHedgehog-Genesis").

## Todo

- NEAT with LSTM's internal state
- Fix and try to replace the LSTM with a MDN LSTM

## Done

- generate data for VAE
- VAE
- generate data for LSTM
- LSTM
- NEAT with VAE (no LSTM)

## To try

- Wasserstein autoencoder.
- Controller + human teaching.
- Transfert learning on the convolutive layers of the VAE : choose a famous network for visual recognition.

## Dependencies

- tensorflow or tensorflow-gpu if you have a recent graphic card (recommended)
- keras
- gym-retro
- numpy
- matplotlib
- pyglet
- pillow
- keyboard
- neat-python
- graphviz (you need to install it on your computer, for windows don't forget to add it to your PATH)
- retrowrapper
