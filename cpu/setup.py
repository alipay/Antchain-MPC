import setuptools

# TODO: we need a license here
setuptools.setup(
    name="morse-stf",
    version="0.1.49",
    author="Morse-STF Team",
    author_email="morse-stf@service.alipay.com",
    description="Morse Secure TensorFlow",
    url="https://github.com/alipay/Antchain-MPC/tree/main/morse-stf",
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
    package_dir={"stensorflow.cops": "./stensorflow/cops/"},
    package_data={"stensorflow.cops": ["*.so"]},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(include=['stensorflow', 'stensorflow.*']) + ["stensorflow.cops"],
    python_requires=">=3.6",
)





# import setuptools
#
# # TODO: we need a license here
# setuptools.setup(
#     name="morse-stf",
#     version="0.1.34",
#     author="Morse-STF Team",
#     author_email="morse-stf@service.alipay.com",
#     description="Morse Secure TensorFlow",
#     url="https://github.com/alipay/Antchain-MPC/morse-stf",
#     install_requires=[
#         'matplotlib==3.3.2',
#         'tensorflow==2.2.0',
#         'pandas==1.0.5',
#         'sympy==1.6',
#         'scikit-learn==0.23.1'
#     ],
#
#
#     entry_points={
#         'console_scripts': [
#             'morse-stf-server=stensorflow.engine.start_server:main',
#         ],
#     },
#     package_dir={"stensorflow.cops": "./cops/"},
#     package_data={"stensorflow.cops": ["*.so"]},
#     classifiers=[
#         "Programming Language :: Python :: 3",
#     ],
#     packages=setuptools.find_packages(include=['stensorflow', 'stensorflow.*']) + ["stensorflow.cops"],
#     python_requires=">=3.6",
# )

