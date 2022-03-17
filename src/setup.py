try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='marie-icr',
    version='1.0.2',
    include_package_data=True,
    description='Python library to Integrate AI-powered OCR features into your applications',
    author='Greg',
    url='https://github.com/gregbugaj/marie-icr',
    packages=['marie-icr'],
    license='Apache License 2.0',
    keywords=['ocr', 'omr', 'optical character recognition','optical mark recognition', 'iteligent character recognition'],
    install_requires=[
        'numpy',
        'Flask>=1.1.1',
        'Flask-Cors>=3.0.8'
    ],
)