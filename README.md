# OrientationField

[![License MIT](https://img.shields.io/pypi/l/orientationfield.svg?color=green)](https://github.com/ChrisMzz/orientationfield/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/orientationfield.svg?color=green)](https://pypi.org/project/orientationfield)
[![Python Version](https://img.shields.io/pypi/pyversions/orientationfield.svg?color=green)](https://python.org)
[![tests](https://github.com/ChrisMzz/orientationfield/workflows/tests/badge.svg)](https://github.com/ChrisMzz/orientationfield/actions)
[![codecov](https://codecov.io/gh/ChrisMzz/orientationfield/branch/main/graph/badge.svg)](https://codecov.io/gh/ChrisMzz/orientationfield)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/orientationfield)](https://napari-hub.org/plugins/orientationfield)

A plugin to compute a nematic field and topological defects on an image.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `orientationfield` via [pip]:

    pip install git+https://github.com/ChrisMzz/OrientationField

## Details

### About nematics : 
Nematics are `(2,2)`-shaped arrays of the form :

$$Q = \begin{pmatrix} Q_{xx} & Q_{xy} \\ 
Q_{xy} & -Q_{xx} \end{pmatrix}$$

representing an orientation in a hemisphere (like arrows without arrowheads).
We have $Q_{xx} = \lvert Q \rvert \cos(2 \psi)$ and $Q_{xy} = \lvert Q \rvert \sin(2 \psi)$, so we can find $\lvert Q \rvert$ and $\psi$ from :

$$\lvert Q \rvert = \sqrt{Q_{xx}^2 + Q_{xy}^2}, \qquad \psi = \frac{1}{2} \arctan 2(Q_{xy}, Q_{xx})$$

We can use builtin SciPy convolutions with kernels $K_{xx}$ and $K_{xy}$, built specifically for this, to get a per-pixel approximation of the nematic field of an image : 

$$K_{xx \vert ij} = e^{-\frac{\rho_{ij}^2}{\sigma^2}} \cos(2 \theta)$$

$$K_{xy \vert ij} = e^{-\frac{\rho_{ij}^2}{\sigma^2}} \sin(2 \theta)$$

For all $i,j$ such that $\rho_{ij} < r$, otherwise the kernels are worth 0.
Here, $\rho_{ij}$ is the distance from the center of the kernel (the kernels are always of shape `(2r+1,2r+1)` where `r` is the radius of a circle defined by a specified parameter), $\theta$ is the angle formed by the position $(i,j)$ compared to the center, and $\sigma$ is a parameter representing kernel bandwidth.

Because of the equivalent amount of information carried by the tuples $(Q_{xx}, Q_{xy})$ and $(\lvert Q \rvert, \psi)$, we can provide both types of data according to different situations.

### Usage 

    Will write more when paper gets closer to being published. For now, installing the package in an environment and running napari in the environment will load the plugin in napari, which can be used with the interface directly.



### Some images

---
Here is what the plugin widget looks like : 

![](https://github.com/ChrisMzz/orientationfield/blob/main/docs/readme_images/widget_ui.png)

---
A brief overview of the parameters :

![](https://github.com/ChrisMzz/orientationfield/blob/main/docs/readme_images/fig3params_exp.png)


---
The tool can be used on non biological data :

![](https://github.com/ChrisMzz/orientationfield/blob/main/docs/readme_images/frctl1.png)
![](https://github.com/ChrisMzz/orientationfield/blob/main/docs/readme_images/frctl2.png)


Here is an example on *The Starry Night* by Van Gogh, this time showcasing defect computation :

![](https://github.com/ChrisMzz/orientationfield/blob/main/docs/readme_images/starrynight_64_hsv.png)

![](https://github.com/ChrisMzz/orientationfield/blob/main/docs/readme_images/starrynight_64_norm.png)




## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"orientationfield" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
