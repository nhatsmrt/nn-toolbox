# A Toolbox for Common Deep Learning Procedures.

## Introduction

Whenever I do a project, I always have to re-implement everything from scratch. At first, this is helpful because it requires me to really learn the concepts and procedures by heart. However, these chores quickly become irritating and annoying. So I start to create this repository, to store all the useful pieces of code.
<br />
This soon extends to the stuff that I see in papers and want to implement. Finally, as I was implementing some of the more difficult stuff (e.g the callbacks) the awesome fastai course comes out, so I decide to use this opportunity to follow along the lessons, adapting the codes (i.e the notebooks) and the library to suit my existing codebase.

## What can you find here?

I organize the codes into core elements (callbacks, components, losses, metrics, optim, transforms, etc.), and applications (vision, sequence, etc.), each having its own elements directory.
## How do I use the codes?

To use the codes here, simply clone this repository and add it to your favourite project. Then do a simple import call, and have fun deep learning!
<br />
Alternatively, you can just select a piece of code that you need and copy it to your project. No need to ask for permission (unless it's something that's not originally mine either, such as the codes adapted from fastai library courses). However, do be aware that the one function might requires another function from another directory to work.
<br />
Note that I have released the codes as a package on PyPI, but right now the everything is still highly experimental and volatile. Nevertheless, you can install it with:
```
pip install nn-toolbox
```

## Some Examples:

I am currently doing some projects with this toolbox. Some of them are still work in progress, but you can visit my [implementation](https://github.com/nhatsmrt/torch-styletransfer) of arbitrary style transfer for some example usage, or look at some [tests](https://github.com/nhatsmrt/nn-toolbox/tree/experimental/nntoolbox/test).

## Documentation

Please visit https://nhatsmrt.github.io/nn-toolbox/
