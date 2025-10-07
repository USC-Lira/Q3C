from setuptools import setup, find_packages

setup(
    name='q3c',
    version='0.1.0',
    description='Actor-Free Continuous Control via Structurally Maximizable Q-Functions',
    author='LiraLab',
    author_email='example@example.com',
    url='https://github.com/USC-Lira/Q3C',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium',
        'torch',
        'stable-baselines3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
)
