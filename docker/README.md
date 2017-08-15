<<<<<<< HEAD
# Caffe standalone Dockerfiles.

The `standalone` subfolder contains docker files for generating both CPU and GPU executable images for Caffe. The images can be built using make, or by running:

```
docker build -t caffe:cpu standalone/cpu
```
for example. (Here `gpu` can be substituted for `cpu`, but to keep the readme simple, only the `cpu` case will be discussed in detail).

Note that the GPU standalone requires a CUDA 7.5 capable driver to be installed on the system and [nvidia-docker] for running the Docker containers. Here it is generally sufficient to use `nvidia-docker` instead of `docker` in any of the commands mentioned.

# Running Caffe using the docker image

In order to test the Caffe image, run:
```
docker run -ti caffe:cpu caffe --version
```
which should show a message like:
```
libdc1394 error: Failed to initialize libdc1394
caffe version 1.0.0-rc3
```

One can also build and run the Caffe tests in the image using:
```
docker run -ti caffe:cpu bash -c "cd /opt/caffe/build; make runtest"
```

In order to get the most out of the caffe image, some more advanced `docker run` options could be used. For example, running:
```
docker run -ti --volume=$(pwd):/workspace caffe:cpu caffe train --solver=example_solver.prototxt
```
will train a network defined in the `example_solver.prototxt` file in the current directory (`$(pwd)` is maped to the container volume `/workspace` using the `--volume=` Docker flag).

Note that docker runs all commands as root by default, and thus any output files (e.g. snapshots) generated will be owned by the root user. In order to ensure that the current user is used instead, the following command can be used:
```
docker run -ti --volume=$(pwd):/workspace -u $(id -u):$(id -g) caffe:cpu caffe train --solver=example_solver.prototxt
```
where the `-u` Docker command line option runs the commands in the container as the specified user, and the shell command `id` is used to determine the user and group ID of the current user. Note that the Caffe docker images have `/workspace` defined as the default working directory. This can be overridden using the `--workdir=` Docker command line option.

# Other use-cases

Although running the `caffe` command in the docker containers as described above serves many purposes, the container can also be used for more interactive use cases. For example, specifying `bash` as the command instead of `caffe` yields a shell that can be used for interactive tasks. (Since the caffe build requirements are included in the container, this can also be used to build and run local versions of caffe).

Another use case is to run python scripts that depend on `caffe`'s Python modules. Using the `python` command instead of `bash` or `caffe` will allow this, and an interactive interpreter can be started by running:
```
docker run -ti caffe:cpu python
```
(`ipython` is also available in the container).

Since the `caffe/python` folder is also added to the path, the utility executable scripts defined there can also be used as executables. This includes `draw_net.py`, `classify.py`, and `detect.py`

=======
### Running an official image

You can run one of the automatic [builds](https://hub.docker.com/r/bvlc/caffe). E.g. for the CPU version:

`docker run -ti bvlc/caffe:cpu caffe --version`

or for GPU support (You need a CUDA 8.0 capable driver and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)):

`nvidia-docker run -ti bvlc/caffe:gpu caffe --version`

You might see an error about libdc1394, ignore it.

### Docker run options

By default caffe runs as root, thus any output files, e.g. snapshots, will be owned
by root. It also runs by default in a container-private folder.

You can change this using flags, like user (-u), current directory, and volumes (-w and -v).
E.g. this behaves like the usual caffe executable:

`docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) bvlc/caffe:cpu caffe train --solver=example_solver.prototxt`

Containers can also be used interactively, specifying e.g. `bash` or `ipython`
instead of `caffe`.

```
docker run -ti bvlc/caffe:cpu ipython
import caffe
...
```

The caffe build requirements are included in the container, so this can be used to
build and run custom versions of caffe. Also, `caffe/python` is in PATH, so python
utilities can be used directly, e.g. `draw_net.py`, `classify.py`, or `detect.py`.

### Building images yourself

Examples:

`docker build -t caffe:cpu cpu`

`docker build -t caffe:gpu gpu`

You can also build Caffe and run the tests in the image:

`docker run -ti caffe:cpu bash -c "cd /opt/caffe/build; make runtest"`
>>>>>>> caffe-bvlc-dev/master
