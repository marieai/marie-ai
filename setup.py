import os
import sys
from os import path

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info < (3, 10, 0):
    raise OSError(f"Marie requires Python >=3.10, but yours is {sys.version}")

try:
    # marie was already taken by PIP
    pkg_name = "marie-ai"
    lib_name = "marie"
    libinfo_py = path.join(lib_name, "__init__.py")
    libinfo_content = open(libinfo_py, "r", encoding="utf8").readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith("__version__")][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError:
    __version__ = "0.0.0"

try:
    with open("README.md", encoding="utf8") as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ""


def register_ac():
    import os
    import re
    from pathlib import Path

    home = str(Path.home())
    resource_path = "marie/resources/completions/marie.%s"
    regex = r"#\sMARIE_CLI_BEGIN(.*)#\sMARIE_CLI_END"
    _check = {"zsh": ".zshrc", "bash": ".bashrc", "fish": ".fish"}

    def add_ac(k, v):
        v_fp = os.path.join(home, v)
        if os.path.exists(v_fp):
            with open(v_fp) as fp, open(resource_path % k) as fr:
                sh_content = fp.read()
                if re.findall(regex, sh_content, flags=re.S):
                    _sh_content = re.sub(regex, fr.read(), sh_content, flags=re.S)
                else:
                    _sh_content = sh_content + "\n\n" + fr.read()

            if _sh_content:
                with open(v_fp, "w") as fp:
                    fp.write(_sh_content)

    try:
        for k, v in _check.items():
            add_ac(k, v)
    except Exception:
        pass


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        register_ac()


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        register_ac()


def get_extra_requires(path, add_all=True):
    import re
    from collections import defaultdict

    try:
        with open(path) as fp:
            extra_deps = defaultdict(set)
            for k in fp:
                if k.strip() and not k.startswith("#"):
                    tags = set()
                    if ":" in k:
                        rpos = k.rindex(":")
                        k, v = (k[0:rpos], k[rpos + 1 : len(k)])
                        print(f"++ {k}  ===== {v}")
                        # k, v = k.split(":")
                        tags.update(vv.strip() for vv in v.split(","))
                    tags.add(re.split("[<=>]", k)[0])
                    for t in tags:
                        extra_deps[t].add(k)

            # add tag `all` at the end
            if add_all:
                extra_deps["all"] = set(vv for v in extra_deps.values() for vv in v)
        return extra_deps
    except FileNotFoundError:
        return {}


all_deps = get_extra_requires("extra-requirements.txt")
core_deps = all_deps["core"]
perf_deps = all_deps["perf"].union(core_deps)
standard_deps = all_deps["standard"].union(core_deps).union(perf_deps)

if os.name == "nt":
    # uvloop is not supported on windows
    exclude_deps = {i for i in standard_deps if i.startswith("uvloop")}
    perf_deps.difference_update(exclude_deps)
    standard_deps.difference_update(exclude_deps)
    for k in ["all", "devel", "cicd"]:
        all_deps[k].difference_update(exclude_deps)

# by default, final deps is the standard deps, unless specified by env otherwise
final_deps = standard_deps

# Use env var to enable a minimum installation of Marie
# MARIE_PIP_INSTALL_CORE=1 pip install marie
# MARIE_PIP_INSTALL_PERF=1 pip install marie
if os.environ.get("MARIE_PIP_INSTALL_CORE"):
    final_deps = core_deps
elif os.environ.get("MARIE_PIP_INSTALL_PERF"):
    final_deps = perf_deps


setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description="Python library to Integrate AI-powered features into your applications",
    author="Marie AI",
    author_email="hello@marieai.co",
    license="Apache 2.0",
    url="https://github.com/marieai/marie-ai/",
    download_url="https://github.com/marieai/marie-ai/tags",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    install_requires=list(final_deps),
    extras_require=all_deps,
    entry_points={
        "console_scripts": [
            "marie=marie_cli:main",
        ],
    },
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Unix Shell",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Documentation": "https://docs.marieai.co",
        "Source": "https://github.com/marieai/marie-ai.git",
        "Tracker": "https://github.com/marieai/marie-ai/issues",
    },
    keywords=(
        "marie-ai ocr icr index elastic neural-network encoding "
        "embedding serving docker container image video audio deep-learning mlops"
    ),
)
