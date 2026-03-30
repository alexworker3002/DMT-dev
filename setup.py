from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "morse_3d",
        ["morse_3d.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14", "-stdlib=libc++"],
    ),
]

setup(
    name="morse_3d",
    ext_modules=ext_modules,
)
