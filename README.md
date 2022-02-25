# GWG-classifier
A convolutional neural network for classifying gravitational-wave glitches from gravitational-wave detectors such as LIGO, Virgo or KAGRA. The intention of this is to train on LIGO glitches and test on [MLy](https://git.ligo.org/mly/mly) glitches. One motivation for this is to have a glitch classification feature added to MLy's functionality.

## Things to do:
1. Optimise CNN architecture
2. Create confusion matrices to see which glitch types the network struggles to recognise
3. Use Keras data augmentation tools to increase the dataset size
4. Await O3 glitch data
5. Further down the line, implement scaleograms
