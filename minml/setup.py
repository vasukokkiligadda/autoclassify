from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="minml",
    version="1.0.0",
    description="A Python package to train all sklearn classifiers on data.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vasukokkiligadda/minml",
    author="vasu kokkiligadda",
    author_email="vasu.kokkiligadda@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=["minml"],
    include_package_data=True,
    install_requires=["re","time","sklearn","pickle","os","mlflow","pycm","inspect","numpy","pandas","matplotlib","seaborn"],
    entry_points={
        "console_scripts": [
            "weather-reporter=weather_reporter.cli:main",
        ]
    },
)