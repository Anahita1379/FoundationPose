DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11
pip install "setuptools<70" wheel
cd /kaolin && rm -rf build *egg* && FORCE_CUDA=1 pip install --no-build-isolation -e .
# Optional: BundleSDF CUDA ops (required for model-free NeRF path only)
# cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd "${DIR}"
