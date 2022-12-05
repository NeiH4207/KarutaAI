from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='',
    version='0.1',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Vu Quoc Hien',
    author_email='hienvq.2000@gmail.com',
    url='https://github.com/NeiH4207/KaturaAI',
    packages=["src", "models", "libraries", "configs"],
    keywords='',
    install_requires=[
        'matplotlib==3.5.1',
        'numpy==1.21.2',
        'pandas==1.3.3',
        'torch==1.9.0',
        'tqdm==4.62.3',
        'torchsummary==1.5.1',
        'scikit-learn==1.0.2',
        'seaborn==0.11.2',
        'librosa==0.8.1'
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'karuta = interaction:main',
        ]
    },

)
