import setuptools
import subprocess
from setuptools.command.install import install
import os

def _copy_cops():
    print("_compile_cops:", os.getcwd())
    home = os.environ['HOME']
    os.makedirs(home + '/stf_cops', exist_ok=True)
    subprocess.check_call('cp -r ./cops/* ~/stf_cops/', shell=True)
    cops_path = home + '/stf_cops/'
    print("cops_path=", cops_path)
    #subprocess.check_call('sh compile.sh', cwd=cops_path, shell=True)


class MorseSTFInstall(install):
    def run(self):
        super().run()
        _copy_cops()


# TODO: we need a license here
setuptools.setup(
    name="morse-stf",
    version="0.1.18",
    author="Morse-STF Team",
    author_email="morse-stf@service.alipay.com",
    description="Morse Secure TensorFlow",
    url="https://github.com/alipay/Antchain-MPC/morse-stf",
    install_requires=[
        'matplotlib==3.3.2',
        'tensorflow==2.2.0',
        'pandas==1.0.5',
        'sympy==1.6',
        'scikit-learn==0.23.1'
    ],


    entry_points={
        'console_scripts': [
            'morse-stf-server=stensorflow.engine.start_server:main',
        ],
    },
    cmdclass={
        'install': MorseSTFInstall,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(include=['stensorflow', 'stensorflow.*']),
    python_requires=">=3.6",
)


#
# install_requires=[
#         'absl-py==0.9.0',
#         'astor==0.8.1',
#         'astunparse==1.6.3',
#         'attrs==19.3.0',
#         'cachetools==4.1.0',
#         'certifi==2020.12.5',
#         'chardet==3.0.4',
#         'click==7.1.2',
#         'coverage==5.1',
#         'cycler==0.10.0',
#         'Cython==0.29.21',
#         'dataclasses==0.8',
#         'dill==0.3.3',
#         'Flask==1.1.2',
#         'future==0.18.2',
#         'gast==0.3.3',
#         'google-auth==1.13.1',
#         'google-auth-oauthlib==0.4.1',
#         'google-pasta==0.2.0',
#         'googleapis-common-protos==1.53.0',
#         'grpcio==1.28.1',
#         'h5py==2.10.0',
#         'idna==2.9',
#         'importlib-metadata==1.7.0',
#         'importlib-resources==5.1.2',
#         'itsdangerous==1.1.0',
#         'Jinja2==2.11.3',
#         'joblib==0.15.1',
#         'Keras-Applications==1.0.8',
#         'Keras-Preprocessing==1.1.0',
#         'kiwisolver==1.2.0',
#         'Markdown==3.2.1',
#         'MarkupSafe==1.1.1',
#         'matplotlib==3.3.2',
#         'mock==4.0.2',
#         'more-itertools==8.4.0',
#         'mpmath==1.1.0',
#         'numpy==1.18.2',
#         'NZMATH==1.1.0',
#         'oauthlib==3.1.0',
#         'opt-einsum==3.2.0',
#         'packaging==20.4',
#         'pandas==1.0.5',
#         'Pillow==8.1.0',
#         'pluggy==0.13.1',
#         'promise==2.3',
#         'protobuf==3.15.5',
#         'psutil==5.8.0',
#         'py==1.9.0',
#         'pyasn1==0.4.8',
#         'pyasn1-modules==0.2.8',
#         'pyparsing==2.4.7',
#         'pytest==5.4.3',
#         'python-dateutil==2.8.1',
#         'pytz==2020.1',
#         'requests==2.23.0',
#         'requests-oauthlib==1.3.0',
#         'rsa==4.0',
#         'scikit-learn==0.23.1',
#         'scipy==1.4.1',
#         'six==1.14.0',
#         'sympy==1.6',
#         'tensorboard==2.1.1',
#         'tensorflow==2.2.0rc0',
#         'tensorflow-datasets==4.2.0',
#         'tensorflow-estimator==2.1.0',
#         'tensorflow-metadata==0.28.0',
#         'termcolor==1.1.0',
#         'threadpoolctl==2.1.0',
#         'torch==1.8.1',
#         'torchtext==0.9.1',
#         'tqdm==4.59.0',
#         'typing-extensions==3.7.4.3',
#         'urllib3==1.25.9',
#         'wcwidth==0.2.5',
#         'Werkzeug==1.0.1',
#         'wrapt==1.12.1',
#         'xmlrunner==1.7.7',
#         'zipp==3.1.0'
#     ],



