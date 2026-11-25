# Data preparation

Download the UCI HAR archive from the [official UCI dataset record](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones), extract it, and run `prepare_data.py` with the extracted `UCI HAR Dataset` folder. The generated `data/har.npz` is ignored by Git.

The preprocessing script consumes the nine `Inertial Signals` channels and the original train/test subject split. It does not redistribute the dataset. See the project README for citation and license notes.
