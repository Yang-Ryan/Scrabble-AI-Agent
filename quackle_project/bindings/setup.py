from setuptools import setup, Extension

module = Extension(
    "_quackle",
    sources=["quackle_wrap.cxx"],
    libraries=["libquackle", "quackleio", "Qt5Core"],
    library_dirs=[
        "/home/kim/quackle/quacker/build/libquackle",
        "/home/kim/quackle/quacker/build/quackleio",
        "/usr/lib/x86_64-linux-gnu"
    ],
    include_dirs=[
        "/home/kim/quackle",
        "/home/kim/quackle/quacker",
        "/home/kim/quackle/quacker/quackleio",
        "/usr/include/x86_64-linux-gnu/qt5",
        "/usr/include/x86_64-linux-gnu/qt5/QtCore",
        "/usr/include"
    ],
    extra_link_args=["-lstdc++"]
)

setup(
    name="quackle",
    ext_modules=[module]
)