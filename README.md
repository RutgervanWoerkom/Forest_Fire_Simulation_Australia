# Modeling Australian forest fire spreading using cellular automata
In past months it was hard to miss, big parts of Australia are on fire threatening nature, humans and wildlife. Altogether, this makes it important and interesting to investigate how wildfires spread. The numerical method used was 2D cellular automata. The model differentiated between the cell states; water, land, vegetation-unaffected, vegetation-freshly lit, vegetation-established fire, vegetation-glowing embers, vegetation-moderately affected, vegetation-strongly affected, and vegetation-ash. The rules for fire to spread to neighbouring cells were based on the following environmental factors:
- Vegetation density
- Height,
- Temperature
- Rain
- Wind

The model was validated by satellite fire data for October, November and December 2019 provided by NASA. For every day in this period, when a fire started in South-East Australia, the model initiated a fire at that specific location.
## Prerequisites
The code is written and tested in:
```python
Python 3.7
```
To run the code make sure the following python packages are installed:
- IPython
- jupyterlab
- numpy
- matplotlib
- ffmpeg-python

If not, the packages can be installed using
```python
pip3 install <package name>
```
## Usage
1. Clone this git repository.
2. Download [this](https://drive.google.com/open?id=1LBQqKTt6GcZKEYzCz8KUSwjUw9FQpF39) .zip file and extract to the folder **datasets/processed**
3. Open **main.ipynb** to configure and run the model.
4. (Optional) Replace environmental factors in **datasets/...** to alter the validation.
## Authors
- Bob Leijnse
- Midas Amersfoort
- Rutger van Woerkom
## Support
For question or help please mail to [bob.leijnse@student.uva.nl](mailto:bob.leijnse@student.uva.nl) for additional information.
## License
[GNU LGPLv3](https://choosealicense.com/licenses/mit)
## Acknowledgments
This project is inspired by:
- http://www.eddaardvark.co.uk/v2/automata/forest.html#picture for providing a framework.
- https://www.sciencedirect.com/science/article/pii/S0304380096019424 for providing a scientific justification.
